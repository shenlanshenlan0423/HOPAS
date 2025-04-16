# -*- coding: utf-8 -*-
"""
@Auth ： Hongwei
@File ：main.py
@IDE ：PyCharm
"""
from src.eval.agent_eval import FQI_for_Q_estimate, KNN_approx_behavior_policy_for_test_data, \
    print_results, run_evaluate, get_Q_values_with_outcome
from src.eval.cluster_eval import cluster_evaluate
from src.model.HOPAS import HOPAS_A, HOPAS_B
from src.model.model_conf import Configs
from src.model.seq2seq import *
from src.model.D3QN import DuelingDoubleDQN
from src.utils.rl_utils import prepare_replay_buffer, training_DQN, testing_DQN
from src.utils.utils import *
from src.data.dataset_split import get_tensor_data


class ExperimentManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def run_train_encoder(self, load_model):
        losses = {}
        for fold_idx in range(folds):
            fold_label = 'fold-{}'.format(fold_idx + 1)
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            if not load_model:
                train_set = prepare_tuples(torch.load(data_path + '/train_set_tuples'))
                val_set = prepare_tuples(torch.load(data_path + '/val_set_tuples'))
                train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)

            if not load_model:
                enc = AttentionEncoder(args.static_dim, args.temporal_dim, args.d, args.embedding_dim,
                                       args.hidden_size, args.num_layers, args.dropout_rate).to(device)
                dec = AttentionDecoder(args.temporal_dim, args.d, args.embedding_dim, args.latent_dim,
                                       args.hidden_size, args.num_layers, args.dropout_rate).to(device)
                model = Seq2Seq(encoder=enc, decoder=dec, static_dim=args.static_dim,
                                hidden_size=args.hidden_size, latent_dim=args.latent_dim).to(device)
                loss = train(model, train_loader=train_loader, valid_loader=val_loader,
                             model_name=args.model_name, fold_label=fold_label, args=args)
                losses[fold_label] = loss
        if not load_model:
            save_pickle(losses, RESULT_DIR + '/{}_losses.pkl'.format(args.model_name))

    def run_phenotypes_extraction(self, load_model):
        trained_cluster_models, cluster_models_eval_res = {}, {}
        for fold_idx in range(folds):
            fold_label = 'fold-{}'.format(fold_idx + 1)
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            trained_encoder = torch.load(MODELS_DIR + '/{}-{}.pt'.format(args.model_name, fold_label))

            train_set = prepare_tuples(torch.load(data_path + '/train_set_tuples'))
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False)

            # 对train data中的所有患者获得latent representation
            patient_level_representations = []
            for (s, a, _, lens, _) in tqdm(train_loader):
                time_mask = torch.zeros((s.shape[0], horizon), dtype=torch.bool)
                for i in range(s.shape[0]):
                    time_mask[i, :lens[i]] = True
                _, attn_represent, _, _, _ = trained_encoder.encoder(s, a, time_mask)
                patient_level_representations.extend(attn_represent.detach().cpu().numpy().tolist())
            patient_level_representations = np.array(patient_level_representations)

            # # ---------------------------------Params selection---------------------------------
            # trained_cluster_models, eval_res = {}, {}
            # for n_clusters in [2, 3, 4, 5, 6, 7, 8, 9]:
            #     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(patient_level_representations)
            #     get_StratificationRes(patient_level_representations, kmeans)
            #     eval_res[n_clusters] = cluster_evaluate(model=kmeans, X=patient_level_representations)
            #     trained_cluster_models[n_clusters] = kmeans
            # save_pickle(trained_cluster_models, MODELS_DIR + 'n_clusters_selection.pkl')
            # save_pickle(eval_res, RESULT_DIR + 'cluster_eval_res_for_elbow.pkl')

            if not load_model:
                # 对患者的表示进行聚类，以提取SIT的典型亚型
                kmeans = KMeans(n_clusters=args.n_clusters, random_state=42).fit(patient_level_representations)
                trained_cluster_models[fold_label] = kmeans
            else:
                trained_cluster_models = load_pickle(MODELS_DIR + '/Cluster_models.pkl')
                kmeans = trained_cluster_models[fold_label]
            cluster_models_eval_res[fold_label] = cluster_evaluate(model=kmeans, X=patient_level_representations)
            if fold_label == 'fold-1':
                save_pickle((patient_level_representations, kmeans), RESULT_DIR + 'res_for_plot_Stratification.pkl')

        if not load_model:
            save_pickle(trained_cluster_models, MODELS_DIR + '/Cluster_models.pkl')
            save_pickle(cluster_models_eval_res, RESULT_DIR + '/Cluster_models_eval_res.pkl')

    def run_rl_data_prepare(self):
        trained_cluster_models = load_pickle(MODELS_DIR + '/Cluster_models.pkl')
        for fold_idx in range(folds):
            fold_label = 'fold-{}'.format(fold_idx + 1)
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            train_set = pd.read_csv(data_path + '/df_train.csv')

            kmeans = trained_cluster_models[fold_label]
            cluster_label = kmeans.labels_  # 获得train set中9348名患者的簇label
            # Fold-1
            # 2    3318
            # 1    3159
            # 0    2871
            gps = train_set.groupby('traj')
            patient_group_idx = list(gps.groups.keys())
            for subclass in range(args.n_clusters):
                patients_index = np.where(cluster_label == subclass)[0]
                sub_df = pd.concat([gps.get_group(patient_group_idx[id_]) for id_ in patients_index]).reset_index(drop=True)
                states, acts, lengths, outcomes = get_tensor_data(sub_df['traj'].unique(), sub_df, outcome_col='rewards_90d')
                torch.save((states, acts, lengths, outcomes), os.path.join(data_path, 'class-{}-tuples'.format(subclass)))

    def run_policy_learning(self, Epochs):
        trained_agents = {}
        args.Epochs = Epochs
        for fold_idx in range(folds):
            fold_label = 'fold-{}'.format(fold_idx + 1)
            trained_fold_agents = {}
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            for subclass in range(args.n_clusters):
                sub_train_tuple = torch.load(os.path.join(data_path, 'class-{}-tuples'.format(subclass)))
                agent = DuelingDoubleDQN(state_dim=state_dim, action_dim=args.action_dim,
                                         hidden_dim=args.Hidden_size, gamma=args.gamma)
                sub_train_replay_buffer = prepare_replay_buffer(tensor_tuple=sub_train_tuple, args=args)
                # torch.save(sub_train_replay_buffer, data_path + 'class-{}-train_replay_buffer'.format(subclass))
                # sub_train_replay_buffer = torch.load(data_path + 'class-{}-train_replay_buffer'.format(subclass))
                agent, loss_dict = training_DQN(agent, sub_train_replay_buffer, args)
                trained_fold_agents['class-{}'.format(subclass)] = agent
            trained_agents[fold_label] = trained_fold_agents
        save_pickle(trained_agents, MODELS_DIR + '/trained_agents.pkl')

    def train_eval_related_model(self, val_flag):
        fqi_models, fqi_training_res, behavior_policys, Q_with_outcomes = {}, {}, {}, {}
        for fold_idx in range(folds):
            fold_label = 'fold-{}'.format(fold_idx + 1)
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            # train set从MIMIC数据中导入
            train_path = DATA_DIR + '/rewards_90d/' + '/{}/'.format(fold_label)
            train_replay_buffer = torch.load(train_path + '/train_set_tuples_replay_buffer')
            if val_flag:
                test_replay_buffer = torch.load(data_path + 'val_set_tuples_replay_buffer')
            else:
                test_replay_buffer = torch.load(data_path + 'test_set_tuples_replay_buffer')

            b_s, b_a, b_r, b_ns, b_d = train_replay_buffer.get_all_samples()
            transition_dict_for_train = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            # Training FQI model for state-action pair value estimation
            # Note: Approximate Model主要起个估计V(s)的作用：估计的Q(s,a)对a不敏感
            fqi_model, training_res = FQI_for_Q_estimate(transition_dict_for_train, args)
            save_pickle(fqi_model, MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label))
            fqi_training_res[fold_label] = training_res

            # KNN approximate for estimating behavior policy in val/test data; about 5 min
            behavior_policy = KNN_approx_behavior_policy_for_test_data(transition_dict_for_train, transition_dict)
            behavior_policys[fold_label] = behavior_policy

            # FQI for state-action pair value estimation in test set
            Q_with_outcomes[fold_label] = get_Q_values_with_outcome(transition_dict, fqi_model=load_pickle(MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label)))

        if val_flag:
            behavior_policy_path = RESULT_DIR + '/{}-val-bps.pkl'.format(args.outcome_label)
            Q_with_outcome_path = RESULT_DIR + '/val-Q_with_outcomes.pkl'
        else:
            behavior_policy_path = RESULT_DIR + '/{}-test-bps.pkl'.format(args.outcome_label)
            Q_with_outcome_path = RESULT_DIR + '/test-Q_with_outcomes.pkl'
        save_pickle(fqi_models, MODELS_DIR + '/RF-FQI.pkl')
        save_pickle(fqi_training_res, RESULT_DIR + '/res_of_RF-FQI.pkl')
        save_pickle(behavior_policys, behavior_policy_path)
        save_pickle(Q_with_outcomes, Q_with_outcome_path)

    def run_rl_baseline(self, Epochs, load_model=False):
        args.Epochs = Epochs
        behavior_policys = load_pickle(RESULT_DIR + '/{}-test-bps.pkl'.format(args.outcome_label))
        model_config = Configs(args)
        for model_name in ['DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'WD3QNE', 'XGB-FQI', 'RF-FQI', 'DQN_CQL']:
            training_res, evaluation_results = {}, {}
            for fold_idx in range(folds):
                fold_label = 'fold-{}'.format(fold_idx + 1)
                print('\n----------------------------{} {}----------------------------\n'.format(model_name, fold_label))
                data_path = self.dataset_path + '/{}/'.format(fold_label)
                if not load_model:
                    train_replay_buffer = torch.load(data_path + '/train_set_tuples_replay_buffer')
                    if model_name in ['DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE']:
                        agent = getattr(model_config, model_name)
                        train_function = getattr(model_config, 'train_{}'.format(model_name[-3:]))
                        agent, training_loss = train_function(agent, train_replay_buffer)
                        torch.save(agent, MODELS_DIR + '/[{}]-{}.pt'.format(fold_label, model_name))
                    elif model_name in ['XGB-FQI']:
                        agent = getattr(model_config, 'XGB_FQI')
                        b_s, b_a, b_r, b_ns, b_d = train_replay_buffer.get_all_samples()
                        transition_dict_for_train = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent, training_loss = agent.fit(transition_dict_for_train, args=args)
                        save_pickle(agent, MODELS_DIR + '/[{}]-{}.pkl'.format(fold_label, model_name))
                    training_res[fold_label] = training_loss
                else:
                    if model_name in ['DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE']:
                        agent = torch.load(MODELS_DIR + '/[{}]-{}.pt'.format(fold_label, model_name))
                    elif model_name in ['XGB-FQI', 'RF-FQI']:
                        agent = load_pickle(MODELS_DIR + '/[{}]-{}.pkl'.format(fold_label, model_name))

                test_replay_buffer = torch.load(data_path + 'test_set_tuples_replay_buffer')
                b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
                transition_dict_for_test = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }

                if model_name in ['DQN', 'DoubleDQN', 'DuelingDQN', 'DuelingDoubleDQN', 'DQN_CQL', 'WD3QNE']:
                    test_function = getattr(model_config, 'test_{}'.format(model_name[-3:]))
                    Q_estimate, agent_policy, _ = test_function(agent, transition_dict_for_test)

                elif model_name in ['XGB-FQI']:
                    Q_estimate, agent_policy = agent.predict(transition_dict_for_test)
                elif model_name in ['RF-FQI']:
                    states = torch.cat(transition_dict_for_test['states'], dim=0).cpu().numpy()
                    action_space = pd.get_dummies(pd.Series(list(range(action_dim)))).values
                    i_t1 = np.hstack((np.repeat(states, action_dim, axis=0), np.tile(action_space, states.shape[0]).T))
                    Q_estimate = agent.predict(i_t1).reshape(states.shape[0], action_dim)
                    agent_policy = np.apply_along_axis(softmax, 1, Q_estimate)
                pass
                fqi_model = load_pickle(MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label))
                evaluation_results[fold_label] = run_evaluate(agent_policy, transition_dict=transition_dict_for_test,
                                                              behavior_policy=behavior_policys[fold_label],
                                                              Q_estimate=Q_estimate,
                                                              fqi_model=fqi_model,
                                                              args=args,
                                                              fold_label=fold_label)
            print_results(evaluation_results, model_name)
            if not load_model:
                save_pickle(training_res, RESULT_DIR + '/res_of_{}.pkl'.format(model_name))
            all_agent_eval_res = load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
            all_agent_eval_res[model_name] = evaluation_results
            save_pickle(all_agent_eval_res, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))

    def run_myModel(self, model_name, val_flag=False):
        trained_agents = load_pickle(MODELS_DIR + '/trained_agents.pkl')
        trained_cluster_models = load_pickle(MODELS_DIR + '/Cluster_models.pkl')
        behavior_policys = load_pickle(RESULT_DIR + '/{}-test-bps.pkl'.format(args.outcome_label))
        eval_res_file = load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
        evaluation_results = {}
        print('\n----------------------------Test {}----------------------------\n'.format(model_name))
        for fold_idx in range(folds):
            if fold_idx + 1 == folds:
                args.plot_flag = True
            fold_label = 'fold-{}'.format(fold_idx + 1)
            print('\n----------------------------{}----------------------------\n'.format(fold_label))
            data_path = self.dataset_path + '/{}/'.format(fold_label)
            if model_name == 'HOPAS-A':
                model = HOPAS_A(data_path=data_path,
                                encoder=torch.load(MODELS_DIR + '/{}-{}.pt'.format(args.model_name, fold_label)),
                                cluster_model=trained_cluster_models[fold_label],
                                agent_pool=trained_agents[fold_label], args=args)
            if model_name == 'HOPAS-B':
                model = HOPAS_B(data_path=data_path,
                                encoder=torch.load(MODELS_DIR + '/{}-{}.pt'.format(args.model_name, fold_label)),
                                cluster_model=trained_cluster_models[fold_label],
                                agent_pool=trained_agents[fold_label], args=args)

            if val_flag:
                test_data = torch.load(data_path + '/val_set_tuples')
                test_replay_buffer = torch.load(data_path + 'val_set_tuples_replay_buffer')
            else:
                test_data = torch.load(data_path + '/test_set_tuples')
                test_replay_buffer = torch.load(data_path + 'test_set_tuples_replay_buffer')
            b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d
            }
            Q_estimate, agent_policy, transition_dict = model.test(test_data, transition_dict)

            evaluation_results[fold_label] = run_evaluate(agent_policy, transition_dict=transition_dict,
                                                          behavior_policy=behavior_policys[fold_label],
                                                          Q_estimate=Q_estimate,
                                                          fqi_model=load_pickle(MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label)),
                                                          args=args,
                                                          fold_label=fold_label)
        print_results(evaluation_results, model_name)
        eval_res_file['{}'.format(model_name)] = evaluation_results
        save_pickle(eval_res_file, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))

    def run_ablation_study(self, Epochs, load_model):
        args.Epochs = Epochs
        behavior_policys = load_pickle(RESULT_DIR + '/{}-test-bps.pkl'.format(args.outcome_label))
        model_config = Configs(args)
        for model_name in ['HOPAS_wo_seq2seq']:
            evaluation_results = {}
            for fold_idx in range(folds):
                fold_label = 'fold-{}'.format(fold_idx + 1)
                print('\n----------------------------{} {}----------------------------\n'.format(model_name, fold_label))
                data_path = self.dataset_path + '/{}/'.format(fold_label)
                if not load_model:
                    agent = getattr(model_config, model_name)
                    df_train = pd.read_csv(data_path + '/df_train.csv')
                    agent = agent.fit(df_train, data_path, args=args)
                    save_pickle(agent, MODELS_DIR + '/[{}]-{}.pkl'.format(fold_label, 'HOPAS_wo_seq2seq'))
                else:
                    agent = load_pickle(MODELS_DIR + '/[{}]-{}.pkl'.format(fold_label, 'HOPAS_wo_seq2seq'))

                test_replay_buffer = torch.load(data_path + 'test_set_tuples_replay_buffer')
                b_s, b_a, b_r, b_ns, b_d = test_replay_buffer.get_all_samples()
                transition_dict_for_test = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                df_test = pd.read_csv(data_path + '/df_test.csv')
                Q_estimate, agent_policy, _ = agent.predict(df_test, transition_dict_for_test)
                fqi_model = load_pickle(MODELS_DIR + '/[{}]-RF-FQI.pkl'.format(fold_label))
                evaluation_results[fold_label] = run_evaluate(agent_policy, transition_dict=transition_dict_for_test,
                                                              behavior_policy=behavior_policys[fold_label],
                                                              Q_estimate=Q_estimate,
                                                              fqi_model=fqi_model,
                                                              args=args,
                                                              fold_label=fold_label)
            print_results(evaluation_results, model_name)
            all_agent_eval_res = load_pickle(RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
            all_agent_eval_res['HOPAS_wo_seq2seq'] = evaluation_results
            save_pickle(all_agent_eval_res, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))


def parse_args():
    parser = argparse.ArgumentParser(description="Specify Params for Testbed")
    #  Note: 'rewards_90d' for MIMIC-IV, 'rewards_icu' for eICU
    parser.add_argument('--outcome_label', type=str, default='rewards_90d')
    parser.add_argument('--seed', type=int, default=0)
    # -----------------------------------------------For Encoder-Decoder-----------------------------------------------
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--static_dim', type=int, default=len(static_cols))
    parser.add_argument('--temporal_dim', type=int, default=len(temporal_cols))
    parser.add_argument('--action_dim', type=int, default=action_dim)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--model_name', type=str, default='seq2seq')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    # --------------------------------------------For Phenotypes Extraction---------------------------------------------
    parser.add_argument('--n_clusters', type=int, default=3)
    # ----------------------------------------------------For Agent-----------------------------------------------------
    parser.add_argument('--Epochs', type=int, default=1)
    parser.add_argument('--Hidden_size', type=int, default=128)
    parser.add_argument('--Batch_size', type=int, default=32)
    parser.add_argument('--Lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_', type=float, default=15)
    parser.add_argument('--plot_flag', type=bool, default=False)
    parser.add_argument('--C_0', type=float, default=-0.025)
    parser.add_argument('--C_1', type=float, default=-0.125)
    parser.add_argument('--C_2', type=float, default=-2.0)
    parser.add_argument('--C_3', type=float, default=0.05)
    parser.add_argument('--C_4', type=float, default=0.05)
    parser.add_argument('--terminal_coeff', type=int, default=15)
    # -----------------------------------------------------For FQI------------------------------------------------------
    parser.add_argument('--max_iteration', type=int, default=100)

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    set_seed(seed=args.seed)
    # -----------------------------------------------Internal Validation------------------------------------------------
    dataset_path = DATA_DIR + '/{}/'.format(args.outcome_label)
    experiment = ExperimentManager(dataset_path=dataset_path)
    experiment.run_train_encoder(load_model=True)
    experiment.run_phenotypes_extraction(load_model=True)
    experiment.run_rl_data_prepare()
    experiment.run_policy_learning(Epochs=3)
    experiment.train_eval_related_model(val_flag=False)  # Get behavior policy

    save_pickle({}, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
    experiment.run_rl_baseline(Epochs=1, load_model=False)
    experiment.run_myModel(model_name='HOPAS-B')  # HeterOgeneous Policy dynAmic fuSion -- HOPAS
    # Ablation study
    experiment.run_ablation_study(Epochs=3, load_model=True)

    # -----------------------------------------------External Validation------------------------------------------------
    args.outcome_label = 'rewards_icu'
    dataset_path = DATA_DIR + '/{}/'.format(args.outcome_label)
    experiment = ExperimentManager(dataset_path=dataset_path)
    experiment.train_eval_related_model(val_flag=False)  # Get behavior policy

    save_pickle({}, RESULT_DIR + '/{}_test-eval_res.pkl'.format(args.outcome_label))
    experiment.run_rl_baseline(Epochs=1, load_model=True)
    experiment.run_myModel(model_name='HOPAS-A')
    args.lambda_ = 9
    experiment.run_myModel(model_name='HOPAS-B')

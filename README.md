# Heterogeneous Policy Dynamic Fusion with Offline Reinforcement Learning for Sepsis-induced Thrombocytopenia Treatment Recommendation
### Hongwei He(1), Yun Li(2,3), Yuan Cao(2,3), Mucan Liu(1,4), Chonghui Guo(1), Hongjun Kang(3)
#### 1 Institute of Systems Engineering, Dalian University of Technology, Dalian, 116024, China.
#### 2 Medical School of Chinese PLA, Beijing, 100853, China.
#### 3 Department of Critical Care Medicine, Chinese PLA General Hospital, Beijing, 100853, China.
#### 4 Department of Information Systems, City University of Hong Kong, Hong Kong, China.
### 2025.04.07

## Dependencies
For a straight-forward use of this code, you can install the required libraries from *requirements.txt*: `pip install -r requirements.txt` 

## Parameters Setting
All parameters involved in the model are listed below:
```
    parser = argparse.ArgumentParser(description="Specify Params for Testbed")
    # 'rewards_90d' for MIMIC-IV, 'rewards_icu' for eICU
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
```
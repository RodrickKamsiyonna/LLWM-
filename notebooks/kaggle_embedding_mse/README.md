# LLWM embedding-MSE training on Kaggle TPU v5e-8

Import `llwm_embedding_mse.ipynb` into Kaggle and select **TPU v5e-8** and **Internet on**. Run all cells. The notebook installs a separate Python 3.12 / PyTorch 2.9 / XLA 2.9 environment, prepares identical local data for all eight ranks, validates it, and launches training in a subprocess. It never initializes XLA in the notebook kernel. The launcher rejects CPU fallback or a topology other than eight TPU ranks.

Settings: compressor15/predictor6, action128, next-token embedding MSE with detached targets, target KL1/lambda3; no SIGReg or memory regression. 6000 updates, global batch131072, context1024, device batch1, accumulation16. Two MPC tasks every1000 updates, horizon20/execute10,100 updates,LR0.001.

Checkpoints: `/kaggle/working/checkpoints`; logs and tokenizer: `/kaggle/working/llwm_artifacts`. Keep output files when saving a notebook version. Interactive session storage is not durable unless saved/exported; save a version to preserve outputs. Kaggle session/quota limits may prevent completing6000 updates in a single session. Checkpoints support continuation, but restore the same tokenizer, complete manifest files and run contract. Use a new ATTEMPT_ID for each relaunch, leaving RUN_ID unchanged for resume, and enable RESUME_LATEST=1 only when checkpoint files are present.

For API publication, set your actual Kaggle username in `kernel-metadata.json`, then use `kaggle kernels push -p notebooks/kaggle_embedding_mse --accelerator TpuV5E8`. Verify the actual accelerator in Kaggle; API allocation can differ from metadata, and the notebook fails if no TPU is present. API credentials must be configured privately; never commit them. The notebook may also be downloaded from GitHub and imported manually.

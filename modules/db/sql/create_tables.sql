CREATE TABLE IF NOT EXISTS optuna_study (
    study_id SERIAL PRIMARY KEY,
    dataset_size INTEGER NOT NULL,
    encoded_dim_min INTEGER NOT NULL,
    encoded_dim_max INTEGER NOT NULL,
    initial_out_channels_min INTEGER NOT NULL,
    initial_out_channels_max INTEGER NOT NULL,
    learning_rate_min DOUBLE PRECISION NOT NULL,
    learning_rate_max DOUBLE PRECISION NOT NULL,
    weight_decay_min DOUBLE PRECISION NOT NULL,
    weight_decay_max DOUBLE PRECISION NOT NULL,
    beta1_min DOUBLE PRECISION NOT NULL,
    beta1_max DOUBLE PRECISION NOT NULL,
    beta2_min DOUBLE PRECISION NOT NULL,
    beta2_max DOUBLE PRECISION NOT NULL,
    momentum_min DOUBLE PRECISION NOT NULL,
    momentum_max DOUBLE PRECISION NOT NULL,
    dampening_min DOUBLE PRECISION NOT NULL,
    dampening_max DOUBLE PRECISION NOT NULL,
    optimizer_idx_min INTEGER NOT NULL,
    optimizer_idx_max INTEGER NOT NULL,
    scheduler_gamma_min DOUBLE PRECISION NOT NULL,
    scheduler_gamma_max DOUBLE PRECISION NOT NULL,
    kl_divergence_lambda_min DOUBLE PRECISION NOT NULL,
    kl_divergence_lambda_max DOUBLE PRECISION NOT NULL,
    epochs_min INTEGER NOT NULL,
    epochs_max INTEGER NOT NULL,
    batch_size_min INTEGER NOT NULL,
    batch_size_max DOUBLE PRECISION NOT NULL,
    best_overall_loss_value DOUBLE PRECISION DEFAULT NULL,
    best_kl_divergence_loss_value DOUBLE PRECISION DEFAULT NULL,
    best_loss_value DOUBLE PRECISION DEFAULT NULL
);
CREATE TABLE IF NOT EXISTS optuna_trial (
    study_id INTEGER NOT NULL,
    trial_id INTEGER NOT NULL,
    state VARCHAR(10) NOT NULL,
    encoded_dim INTEGER NOT NULL,
    initial_out_channels INTEGER NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    weight_decay DOUBLE PRECISION NOT NULL,
    beta1 DOUBLE PRECISION NOT NULL,
    beta2 DOUBLE PRECISION NOT NULL,
    momentum DOUBLE PRECISION NOT NULL,
    dampening DOUBLE PRECISION NOT NULL,
    optimizer_idx DOUBLE PRECISION NOT NULL,
    scheduler_gamma DOUBLE PRECISION NOT NULL,
    kl_divergence_lambda DOUBLE PRECISION NOT NULL,
    epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    loss_function VARCHAR(30) NOT NULL,
    overall_loss_value DOUBLE PRECISION NOT NULL,
    kl_divergence_loss_value DOUBLE PRECISION NOT NULL,
    loss_value DOUBLE PRECISION NOT NULL,
    CONSTRAINT id PRIMARY KEY (study_id, trial_id)
);
CREATE TABLE IF NOT EXISTS loss_functions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimizers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
INSERT INTO loss_functions(id, name)
VALUES (0, 'MSELoss'),
    (1, 'RMSELoss') ON CONFLICT DO NOTHING;
INSERT INTO optimizers(id, name)
VALUES (0, 'Adam'),
    (1, 'SGD') ON CONFLICT DO NOTHING;
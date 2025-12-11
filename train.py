from src.core.trainer import Trainer
from src.core.aws_rec_dataset import AwsRecDataset
from src.core.rec_model import SequentialRecommendation

# IV
model = SequentialRecommendation(num_items=7957)
data_train = AwsRecDataset("./data/IV", "train")
data_tests = {
    "test": AwsRecDataset("./data/IV", "test"),
}
trainer = Trainer(
    epoch_num=100,
    train_batch_size=32,
    test_batch_size=1,
    lr=0.001,
)
trainer.setup(
    model=model,
    data_train=data_train,
    data_tests=data_tests,
    log_dir="./exp_IV/logs",
)
trainer.train(checkpoint_dir="./exp_IV/models")

# PS
model = SequentialRecommendation(num_items=33798)
data_train = AwsRecDataset("./data/PS", "train")
data_tests = {
    "test": AwsRecDataset("./data/PS", "test"),
}
trainer = Trainer(
    epoch_num=100,
    train_batch_size=32,
    test_batch_size=1,
    lr=0.001,
)
trainer.setup(
    model=model,
    data_train=data_train,
    data_tests=data_tests,
    log_dir="./exp_PS/logs",
)
trainer.train(checkpoint_dir="./exp_PS/models")

# THI
model = SequentialRecommendation(num_items=66710)
data_train = AwsRecDataset("./data/THI", "train")
data_tests = {
    "test": AwsRecDataset("./data/THI", "test"),
}
trainer = Trainer(
    epoch_num=75,
    train_batch_size=32,
    test_batch_size=1,
    lr=0.001,
)
trainer.setup(
    model=model,
    data_train=data_train,
    data_tests=data_tests,
    log_dir="./exp_THI/logs",
)
trainer.train(checkpoint_dir="./exp_THI/models")

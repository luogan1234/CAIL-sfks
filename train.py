from config import Config
from data_loader import SKFSDataLoader
from processor import Processor
import argparse
import os

def train(args):
    config = Config(args.encoder, True)
    path = 'data/'
    data_loader = SKFSDataLoader(config, path)
    processor = Processor(config, data_loader)
    processor.train()

def test(args):
    config = Config(args.encoder, False)
    path = 'data/'
    data_loader = SKFSDataLoader(config, path)
    processor = Processor(config, data_loader)
    processor.test('result/predict.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFKS')
    parser.add_argument('-encoder', type=str, default='bilstm', choices=['textcnn', 'bilstm', 'bert', 'bert_freeze'])
    args = parser.parse_args()
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/model_states/'):
        os.mkdir('result/model_states/')
    train(args)
    test(args)
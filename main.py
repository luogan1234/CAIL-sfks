from config import Config
from data_loader import SKFSDataLoader
from processor import Processor

if __name__ == '__main__':
    config = Config('bilstm', False)
    path = '/input/'
    data_loader.append(SKFSDataLoader(config, path))
    processor = Processor(config, data_loader)
    processor.test('/output/result.txt')
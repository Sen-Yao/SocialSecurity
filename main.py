import argparse
import torch
import time

import preprocess
import net


def main():
    parser = argparse.ArgumentParser(description="A script with command line arguments.")

    parser.add_argument("--raw_data_path", "-p", help="Input file path")
    parser.add_argument("--learning_rate", "-lr", help="learning rate")
    parser.add_argument("--epoch", "-e", help="epoch")
    parser.add_argument("--batch_size", "-b", help="batch size")

    args = parser.parse_args()
    raw_data_list = preprocess.read_csv_to_dict_list(args.raw_data_path)
    preprocess.clean_invalid_line(raw_data_list)
    X = []
    Y = []
    for data_dict in raw_data_list:
        data_line = []
        for key, value in data_dict.items():
            if key != 'pce':
                data_line.append(float(value))

            else:
                Y.append(torch.tensor(float(value), dtype=torch.float32))
        data_line = torch.tensor(data_line, dtype=torch.float32)
        X.append(data_line)
    predict_net = net.ConsumePredictNet(args.learning_rate)
    t1 = time.time()
    t2 = t1
    try:
        predict_net.net = torch.load('model\\model.pkl')
    except FileNotFoundError:
        print('File Not Found')
    predict_net.net.train()
    for epoch in range(int(args.epoch)):
        # train
        batch_x, batch_y = next(preprocess.get_batch(X, Y, int(args.batch_size)))
        batch_x = torch.tensor(batch_x).to(net.try_gpu())
        batch_y = torch.tensor(batch_y).to(net.try_gpu())
        batch_y = batch_y.view(-1, 1)
        pred_y = predict_net.net(batch_x)
        predict_net.Update(pred_y, batch_y)
        if time.time()-t2 > 5:
            print('saved')
            torch.save(predict_net.net, 'model\\model.pkl')
            t2 = time.time()
        elif time.time()-t1 > 1:
            print('loss=', predict_net.loss.item())
            t1 = time.time()



if __name__ == "__main__":
    main()

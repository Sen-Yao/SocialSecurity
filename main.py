import argparse
import os.path

import torch
import time

import preprocess
import net

VALID_SET = 2500
TEST_SET = 200


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
            if key != 'ln_pce':
                data_line.append(float(value))

            else:
                # print(data_dict)
                Y.append(torch.tensor(float(value), dtype=torch.float32))
        data_line = torch.tensor(data_line, dtype=torch.float32)
        X.append(data_line)
    predict_net = net.ConsumePredictNet(args.learning_rate)
    train_x = X[:VALID_SET]
    train_y = Y[:VALID_SET]
    test_x = X[VALID_SET:VALID_SET+TEST_SET]
    test_y = Y[VALID_SET:VALID_SET + TEST_SET]
    ref_x = X[VALID_SET+TEST_SET:]
    ref_y = Y[VALID_SET + TEST_SET:]
    t1 = time.time()
    t2 = t1
    try:
        predict_net.net = torch.load(os.path.join('model', "model.pkl"))
        predict_net.trainer = torch.optim.Adam(predict_net.net.parameters(), lr=predict_net.learning_rate)
    except FileNotFoundError:
        print('File Not Found')
    predict_net.net.train()
    for epoch in range(int(args.epoch)):
        # train
        predict_net.net.train()
        # dynamic learning rate
        predict_net.learning_rate = float(args.learning_rate) / (epoch + 1)
        predict_net.trainer = torch.optim.Adam(predict_net.net.parameters(), lr=predict_net.learning_rate)

        batch_x, batch_y = next(preprocess.get_batch(train_x, train_y, int(args.batch_size)))
        batch_x = torch.tensor(batch_x).to(net.try_gpu())
        batch_y = torch.tensor(batch_y).to(net.try_gpu())
        batch_y = batch_y.view(-1, 1)
        batch_y += 0.5 * torch.rand_like(batch_y)
        pred_y = predict_net.net(batch_x)
        predict_net.Update(pred_y, batch_y)
        if time.time()-t2 > 30:
            print('saved')
            torch.save(predict_net.net, os.path.join('model', "model.pkl"))
            t2 = time.time()
        elif time.time()-t1 > 1:
            # test
            batch_x, batch_y = next(preprocess.get_batch(test_x, test_y, 10))
            batch_x = torch.tensor(batch_x).to(net.try_gpu())
            batch_y = torch.tensor(batch_y).to(net.try_gpu())
            batch_y = batch_y.view(-1, 1)
            predict_net.net.eval()
            pred_y = predict_net.net(batch_x)
            print('epoch=', epoch, 'loss=', predict_net.loss.item(), 'test bias = ', abs(batch_y-pred_y).mean().item())
            t1 = time.time()
    # 14 is ifsocial, 15 is social_rate_h
    predict_net.net.eval()
    for line_x, origin_y in zip(ref_x, ref_y):
        if line_x[14]:
            line_x[14] = 0
            line_x[15] = 0
            pred_y = predict_net.net(torch.tensor(line_x, dtype=torch.float32, device=net.try_gpu()))
            print('真实数据中，缴纳社保的 pce 为', origin_y.item(), '若不缴纳社保，预测 pce 为', pred_y.item())



if __name__ == "__main__":
    main()

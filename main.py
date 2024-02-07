import argparse
import os.path
import torch
import time
import math

import preprocess
import net

TRAIN_SET = 2400
VALID_SET = 200


def main():
    parser = argparse.ArgumentParser(description="A script with command line arguments.")
    parser.add_argument("--raw_data_path", "-p", help="Input file path")
    parser.add_argument("--learning_rate", "-lr", help="learning rate")
    parser.add_argument("--epoch", "-e", help="epoch")
    parser.add_argument("--batch_size", "-b", help="batch size")
    parser.add_argument("--gamma", "-g", help="gamma")

    args = parser.parse_args()
    raw_data_list, city_code_index = preprocess.read_csv_to_dict_list(args.raw_data_path)
    preprocess.clean_invalid_line(raw_data_list)
    X = []
    Y = []
    for data_dict in raw_data_list:
        data_line = []
        for key, value in data_dict.items():
            if key != 'ln_pce':
                if key == 'citycode2':
                    city_code_idx = city_code_index[int(value)]
                    data_line.append(city_code_idx)
                else:
                    data_line.append(float(value))
            else:
                Y.append(float(value))
        X.append(data_line)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    train_x = X[:TRAIN_SET]
    train_y = Y[:TRAIN_SET]
    test_x = X[TRAIN_SET:TRAIN_SET + VALID_SET]
    test_y = Y[TRAIN_SET:TRAIN_SET + VALID_SET]
    ref_x = X[TRAIN_SET + VALID_SET:]
    ref_y = Y[TRAIN_SET + VALID_SET:]
    train_iter = preprocess.load_array((train_x, train_y), int(args.batch_size))
    valid_iter = preprocess.load_array((test_x, test_y), VALID_SET)
    t1 = time.time()
    t2 = t1
    t3 = t1
    try:
        print(os.path.join('model', "model.pkl"))
        predict_net = torch.load(os.path.join('model', "model.pkl"))
        print('model loaded')
    except FileNotFoundError:
        print('File Not Found, creating a new model')
        predict_net = net.ConsumePredictNet(args.learning_rate, float(args.gamma), len(city_code_index), embedding_dim=10)
    predict_net.train()
    for epoch in range(int(args.epoch)):
        # train
        predict_net.avg_loss = 0
        predict_net.update_in_epoch = 0
        predict_net.train()
        for batch_x, batch_y in train_iter:
            batch_x = batch_x.to(net.try_gpu())
            batch_y = batch_y.to(net.try_gpu())
            batch_y = batch_y.view(-1, 1)
            # Add noise
            batch_y += 0.1 * torch.rand_like(batch_y)
            pred_y = predict_net(batch_x)
            predict_net.Update(pred_y, batch_y)
        predict_net.scheduler.step()

        # valid
        predict_net.eval()

        if time.time()-t1 > 1:
            # test
            batch_x, batch_y = next(iter(valid_iter))
            batch_x = batch_x.to(net.try_gpu())
            batch_y = batch_y.to(net.try_gpu())
            batch_y = batch_y.view(-1, 1)
            pred_y = predict_net(batch_x)
            predict_net.output(pred_y, batch_y)

            t1 = time.time()
        if time.time()-t3 > 10:
            # test
            batch_x, batch_y = next(iter(valid_iter))
            batch_x = batch_x.to(net.try_gpu())
            batch_y = batch_y.to(net.try_gpu())
            batch_y = batch_y.view(-1, 1)
            predict_net.eval()
            pred_y = predict_net(batch_x)
            for line_num in range(len(batch_x)):
                if batch_x[line_num][11] == 1:
                    # print('真实值应为', batch_y[line_num].item(), '预测值为', pred_y[line_num].item(), '差值为', pred_y[line_num].item() - batch_y[line_num].item())
                    pass
            t3 = time.time()

        if time.time()-t2 > 300:
            print('\n\n\nsaved')
            torch.save(predict_net, os.path.join('model', "model.pkl"))
            t2 = time.time()
        predict_net.epoch += 1
    print('训练结束，测试集中：')
    predict_net.eval()
    for line_x, origin_y in zip(ref_x, ref_y):
        line_x = line_x.reshape(1, -1).to(net.try_gpu())
        pred_y = predict_net(line_x)
        print(f'原始数据为{origin_y.item():.4f}, 模型输出为{pred_y.item():.4f}, 偏差为{pred_y.item() - origin_y.item():.4f}')
        # if line_x[11] == 1:
        #     line_x[11] = 0
        #     line_x = line_x.reshape(1, -1).to(net.try_gpu())
        #     pred_y = predict_net(line_x)
        #     print('真实数据中，缴纳社保的 pce 为', origin_y.item(), '若不缴纳社保，预测 pce 为', pred_y.item())
        # else:
        #     line_x[11] = 1
        #     line_x = line_x.reshape(1, -1).to(net.try_gpu())
        #     pred_y = predict_net(line_x)
        #     print('真实数据中，不缴纳社保的 pce 为', origin_y.item(), '若缴纳社保，预测 pce 为', pred_y.item())



if __name__ == "__main__":
    main()

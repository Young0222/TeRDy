import argparse
from typing import Dict
import torch
from torch import optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, TKBCOptimizer_sparse
from models import TCompoundE, TeRDy
from regularizers import N3, Lambda3
import os
from torch.optim.lr_scheduler import LambdaLR


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

parser = argparse.ArgumentParser(
    description="TeRDy"
)
parser.add_argument(
    '--dataset', type=str, default='ICEWS14',
    help="Dataset name"
)

parser.add_argument(
    '--model', default='TeRDy', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=2000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--gpu', default=0, type=int,
    help="GPU."
)
parser.add_argument(
    '--optimizer', default='Adagrad', type=str,
    help="Optimizer Name"
)
parser.add_argument(
    '--freq_reg', default=0., type=float,
    help="Frequency regularizer strength"
)
parser.add_argument(
    '--alpha', default=10, type=float,
    help="Alpha"
)

args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          freq_reg=args.freq_reg
         ):


    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset


    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}/freq_reg{:.5f}/'.format(rank,learning_rate,batch_size, emb_reg, time_reg, freq_reg))
    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    dataset = TemporalDataset(dataset, device)
    
    model = {
        'TeRDy': TeRDy(sizes, rank, no_time_emb=args.no_time_emb, alpha=args.alpha)
    }[model]

    model = model.to(device)

    if args.optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == "AdamW":
        opt = optim.AdamW(model.parameters(), lr=learning_rate)
    elif args.optimizer == "RMSprop":
        opt = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
    elif args.optimizer == "SGD":
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif args.optimizer == "NAdam":
        opt = optim.NAdam(model.parameters(), lr=learning_rate)
    else:
        opt = optim.Adagrad(model.parameters(), lr=learning_rate)

    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg, "freq_reg =", freq_reg)

    emb_reg = N3(emb_reg)
    time_reg = Lambda3(time_reg)
  
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt, freq_reg,
            batch_size=batch_size
        )

        optimizer.epoch(examples, device)   # traing model
       
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

            if dataset.interval: 
                valid, test = [
                    avg_both(*dataset.eval(model, split, -1))
                    for split in ['valid', 'test']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results
            f = open(os.path.join(PATH, 'result.txt'), 'a+')
            f.write("\n[Epoch:{}]-VALID : ".format(epoch))
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            if not dataset.interval:
                curve['train'].append(train)
    
                print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
            print("\t Current TEST : ", test)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'a+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':
    learn()



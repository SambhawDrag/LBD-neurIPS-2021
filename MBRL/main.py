import torch
from torch.utils.data.dataset import Dataset
from models import mlp
import os
import argparse
from util import seed_everything, create_dataloaders, train, test, create_test_loader
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter


def main(args, logger: SummaryWriter):

    # Create dataloaders
    train_loader, val_loaders, dims = create_dataloaders(
        system=args.system,
        root_dir=args.root_dir,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    # Create model
    device = "cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu"
    model = mlp(*dims).to(device)

    # Training Loop
    for epoch in range(args.epochs):
        train_losses = train(model, train_loader, epoch, args)
        logger.add_scalar('train_loss', sum(
            train_losses) / len(train_losses), epoch)

        val_loss = test(model, val_loaders, args)
        logger.add_scalar('val_loss', val_loss, epoch)

        if args.verbose:
            print(tabulate([[epoch, sum(train_losses) / len(train_losses), val_loss]],
                  headers=['Epoch', 'Train Loss', 'Val Loss'], tablefmt='github'))

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(),
                       f'{args.save_dir}/{args.system}_{epoch}_{val_loss}.ckpt')

        if args.fast_dev_run:
            break

    # Testing Loop
    test_loader = create_test_loader(
        system=args.system,
        root_dir=args.root_dir,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_loss = test(model, test_loader, args)
    print(f'Test Loss: {test_loss}')

    torch.save(model.state_dict(), f'{args.save_dir}/{args.system}_final.ckpt')
    logger.add_hparams(vars(args), metric_dict={'test_loss': test_loss})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str,
                        default='great-piquant-bumblebee')
    parser.add_argument('--root_dir', type=str, default='./dataset')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true',
                        help="Use --cuda to use GPU")
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fast_dev_run', action='store_true',
                        help="Runs through all three datasets for a single batch (for debugging)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print training progress")

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    seed_everything(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    writer = SummaryWriter(log_dir=args.log_dir)

    main(args, writer)

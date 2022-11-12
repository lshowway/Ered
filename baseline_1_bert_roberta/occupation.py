import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using %s"%device)

    t1 = torch.ones((100, 1000))
    t2 = torch.ones((100, 1000))

    while True:
        t3 = torch.sum(torch.matmul(t1, t2.transpose(1, 0)))


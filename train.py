from nets.yolo import YoloBody


if __name__ == "__main__":


    # ========================================一些训练设置========================================
    Cuda = True
    

    # 实例化模型，设置优化器
    model     = YoloBody()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(Your_model.parameters(),lr=args.learning_rate)
    
    train_loss        = []
    valid_loss        = []
    train_epochs_loss = []
    valid_epochs_loss = []
    
    early_stopping = EarlyStopping(patience=args.patience,verbose=True)
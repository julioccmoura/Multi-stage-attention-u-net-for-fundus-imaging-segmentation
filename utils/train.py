from tqdm import tqdm

import torch


from utils.evalmetrics import dice_coef, classification_metrics

def train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                    save_path, n_epochs=25, patience=5, stage=""):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_dice, wait = -1.0, 0

    # Histories
    tr_loss_hist, va_loss_hist = [], []
    tr_dice_hist, va_dice_hist = [], []
    

    for epoch in range(1, n_epochs+1):
        model.train()
        running_loss, running_dice = 0.0, 0.0


        for x, y in tqdm(train_loader, desc=f"[{stage}] Epoch {epoch}/{n_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coef(out, y).item()


        # Train metrics
        tr_loss = running_loss / len(train_loader)
        tr_dice = running_dice / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_dice = 0.0, 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                
                val_loss += criterion(out, y).item()
                val_dice += dice_coef(out, y).item()

                all_preds.append(out.cpu())
                all_targets.append(y.cpu())
        
        # Validation metrics
        va_loss = val_loss / len(val_loader)
        va_dice = val_dice / len(val_loader)


        # Store history
        tr_loss_hist.append(tr_loss); va_loss_hist.append(va_loss)
        tr_dice_hist.append(tr_dice); va_dice_hist.append(va_dice)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        acc, specificity, rec, prec, f1,  y_true, y_pred = classification_metrics(all_targets, all_preds)


        # Logging
        print(f"[{stage}] Epoch {epoch}: "
              f"TrainLoss={tr_loss:.4f} ValLoss={va_loss:.4f} "
              f"TrainDice={tr_dice:.4f} ValDice={va_dice:.4f} "
              )

        # Early Stopping
        if va_dice > best_dice:
            best_dice, wait = va_dice, 0
            torch.save(model.state_dict(), save_path)
            print(f"   🔹 New best ValDice = {best_dice:.4f} (saved).")
        else:
            wait += 1
            if wait >= patience:
                print("   ⏹ Early stopping.")
                break
        
    return {
        "train_loss": tr_loss_hist, "val_loss": va_loss_hist,
        "train_dice": tr_dice_hist, "val_dice": va_dice_hist,
        "best_dice": best_dice, 
        "acc": acc,
        "spec": specificity,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }

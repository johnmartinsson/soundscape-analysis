import tqdm

def evaluate_soundscape_source_separation_model(model, valid_loader):
    model.eval()

    accumulated_loss = 0
    count = 0

    for mix, anthropophony, geophony, biophony in valid_loader:
        pred_anthropophony, pred_geophony, pred_biophony = model(mix)
        anthropophony_loss = loss_function(anthropophony, pred_anthropophony)
        geophony_loss = loss_function(geophony, pred_geophony)
        biophony_loss = loss_function(biophony, pred_geophony)
        total_loss = anthropophony_loss + geophony_loss + biophony_loss

        accumulated_loss += total_loss.item()
        count += 1

    return accumulated_loss / count

def train(model, optimizer, loss_function, train_loader, valid_loader, cfg):
    model.train()

    epochs = int(cfg['epochs'])

    for epoch in range(epochs):

        accumulated_loss = 0
        count = 0

        for mix, anthropophony, geophony, biophony in tqdm.tqdm(train_loader):
            print(mix.shape)
            print(anthropophony.shape)
            print(geophony.shape)
            print(biophony.shape)
            optimizer.zero_grad()

            pred_anthropophony, pred_geophony, pred_biophony = model(mix)
            anthropophony_loss = loss_function(anthropophony, pred_anthropophony)
            geophony_loss = loss_function(geophony, pred_geophony)
            biophony_loss = loss_function(biophony, pred_geophony)
            total_loss = anthropophony_loss + geophony_loss + biophony_loss

            total_loss.backward()
            optimizer.step()

            accumulated_loss += total_loss.item()

        train_loss = accumulated_loss / count
        validation_loss = evaluate_soundscape_source_separation_model(model, valid_loader)
        print("train loss: {}, epoch: {}".format(train_loss, epoch))

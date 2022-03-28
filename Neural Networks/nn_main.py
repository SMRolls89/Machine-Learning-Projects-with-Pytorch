import torch
import numpy as np
from matplotlib import pyplot as plt
from nn_models import *

"""
Parameters
"""
num_iter = 100
learning_rate = 0.0005
batch_size = 32

"""
Read data from the specified training, validation and test data files.
We are using the whole image, not creating other features now
"""
def read_data(trainFile, valFile, testFile):
    # trian, validation, and test data loader
    data_loaders = []

    # read training, test, and validation data
    for file in [trainFile, valFile, testFile]:
        # read data
        data = np.loadtxt(file)
        # digit images
        imgs = torch.tensor(data[:,:-1]).float()
        # divide each image by its maximum pixel value for numerical stability
        imgs = imgs / torch.max(imgs,dim=1).values[:,None]

        # labels for each image
        labels = torch.tensor(data[:,-1]).long()

        # if using CNN model, reshape each image:
        # [batch x num_channel x image width x image height]
        if not use_mlp:
            imgs = imgs.view(-1,1,28,28)

        # create dataset and dataloader, a container to efficiently load data in batches
        dataset = utils.TensorDataset(imgs,labels)
        dataloader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_loaders.append(dataloader)
    
    return data_loaders[0], data_loaders[1], data_loaders[2]

"""
Train Multilayer Perceptron (MLP)
Initialize your MLP model --> define loss function --> define optimizer
--> train your model with num_iter epochs --> pick the best model and return
    - Parameters:   train_loader --- the train dataloader
                    val_loader --- the validation dataloader
    - Return:       net --- the best trained MLP network with the lowest validation loss
                    avg_train_loss --- a list of averaged training loss of length num_iter
                    avg_val_loss --- a list of averaged validation loss of length num_iter
"""
def trainMLP(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []
    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []
    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
    
    """
    to test with the baseline nlp model
    net = BaselineMLP()

    to test with the new custom design
    net = my_network()
    
    """
        net = BaselineMLP()

    # define the loss function as crossentropy loss since this is a multi class classification model
    loss_fn = torch.nn.CrossEntropyLoss()
    # define the Adam optimizer which is more robust method with reguralizations inbuid
    optimizer = torch.optim.Adam( net.parameters() , lr= learning_rate )  # define with model parameters and learning_rate for optimization
    
    # implement training and early stopping
    # model early stopping perform if the validation loss does not improve for 5 epochs continously
    early_stopping_margin = 5
    
    # for each iteration, iteratively train all batches
    i = 0
    while i < num_iter:
       
        # increment i
        i += 1

        train_loss ,  iter_train = 0, 0
        # iterate over every training batch using train data loader
        for batch_img , batch_target in train_loader :

            # forward pass the image batch
            model_out = net( batch_img )
            # since the final softmax layer is not embedded into the model, use it before feeding into the loss function
            softmax_out = torch.nn.functional.softmax( model_out , dim = -1 )

            # compute the loss using the defined loss object with model prediction and target labels
            loss_batch = loss_fn( softmax_out , batch_target )

            # clean the optimizer function before optimizing on new lost
            optimizer.zero_grad()

            # compute the backprogation
            loss_batch.backward()

            # optimize the model using optimizer  update the weights on calculated gradient
            optimizer.step()

            # append the batch loss into a array to check later
            train_loss += loss_batch.item()
            # track the number of iteration ran
            iter_train += 1

        # define epoch train loss
        epoch_loss = float( train_loss  ) / float( iter_train )

        # append the epoch loss and accuracy into a list
        avg_train_loss.append( epoch_loss )

        # model validation on validation dataset
        with torch.no_grad():
            # define supportive parameters
            valid_loss  , iter_valid  = 0 , 0 
            # make the model to validation formati
            net.eval()

            # iterate over validation dataloader to take batch of images and labels
            for batch_img , batch_target in val_loader :

                # forward pass the image batch
                model_out = net( batch_img )
                # since the final softmax layer is not embedded into the model use it before feeding into the loss function
                softmax_out = torch.nn.functional.softmax( model_out , dim = -1 )

                # compute the loss using the defined loss object with model prediction and target labels
                loss_batch = loss_fn( softmax_out , batch_target )

                # append the batch loss into an array to check later
                valid_loss += loss_batch.item()
                # track the number of iteration ran
                iter_valid += 1

            # define epoch valid loss
            epoch_loss = float( valid_loss  ) / float( iter_valid )

            # append the epoch validation loss and accuracy into a list
            avg_val_loss.append( epoch_loss )

            net.train()
        
        # save the best model with lowest validation loss and load it to do testing
        # check the validation loss with the best validation loss so far
        if( np.round( best_val_score , 2 ) > np.round( epoch_loss , 2 )  ):
            best_val_score = epoch_loss
            # define a new model instance from randomly initialized weights
            """
            to test with the baseline nlp model
            best_model = BaselineMLP()

            to test with the new custom design
            best_model = my_network()
            
            """
            best_model = BaselineMLP()

            # assign the best validation loss model weights to the randomly initialized weights
            best_model.load_state_dict( net.state_dict().copy()  )
            # make the model evaluation mode
            best_model.eval()

            # reset the early stopping margin
            early_stopping_margin = 0

        # if the current batch loss greater than best valid loss then increment the early stopping count
        elif( np.round( epoch_loss , 2 )  > np.round( best_val_score , 2 ) ):
            early_stopping_margin += 1

        # if the early stopping count goes beyond 5 then stop model traning and validating
        if( early_stopping_margin >= 5 ):
            # stop model traning
            break

        print("Model Training Loss : {:.3f}  Validation Loss : {:.3f}  ".format(
            avg_train_loss[-1] , avg_val_loss[-1] ))
    
    # get the best saved model
    net = best_model
        
    return net, avg_train_loss, avg_val_loss

"""
Train your Baseline Convolutional Neural Network (CNN)
Initialize your CNN model --> define loss function --> define optimizer
--> train your model with num_iter epochs --> pick the best model and return
    - parameters:   train_loader --- the train dataloader
                    val_loader --- the validation dataloader
    - return:       net --- the best trained CNN network with the lowest validation loss
                    train_loss --- a list of training loss
"""
def trainCNN(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []
    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []
    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
    net = BaselineCNN()
    #       define loss function
    #       define optimizer
    #       for each epoch, iteratively train all batches
    i = 0
    while i < num_iter:
        # implement your training and early stopping
        # save the best model with lowest validation loss and load it to do testing
        raise NotImplementedError
    
    return net, avg_train_loss, avg_val_loss


"""
Evaluate the model, using unseen data features "X" and
corresponding labels "y".
Parameters: loader --- the test loader
            net --- the best trained network
Return: the accuracy on test set
"""
def evaluate(loader, net):
    total = 0
    correct = 0
    # use model to get predictions
    for X, y in loader:
        outputs = net(X)
        predictions = torch.argmax(outputs.data, 1)
        
        # total number of items in dataset
        total += y.shape[0]

        # number of correctly labeled items in dataset
        correct += torch.sum(predictions == y)

    # return fraction of correctly labeled items in dataset
    return float(correct) / float(total)

if __name__ == "__main__":

    # test CNN model
    use_mlp = True

    # load data from file
    train_loader, val_loader, test_loader = \
        read_data('train.txt','validate.txt', 'test.txt')

    if use_mlp:
        net, t_losses, v_losses = trainMLP(train_loader, val_loader)
    else:
        net, t_losses, v_losses = trainCNN(train_loader,val_loader)

    # evaluate model on validation data
    accuracy = evaluate(test_loader, net)

    print("Test accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.legend(["training_loss","validation_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.show()
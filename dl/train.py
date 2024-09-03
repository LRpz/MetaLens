import sys
from MetaLens.dl.utils import define_transforms, train_regressor

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: train.py <training_data_folder> <model_folder>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    model_path = sys.argv[2]

    MODEL = 'resnet152'
    patch_size = 128
    training_patch_size = 128 # custom patch size for training using scaling in data augmentation 
    
    batch_size=32
    learning_rate=1e-3
    epochs=200
    encoder=MODEL
    in_chans=4 
    test_size=0.3

    transforms = define_transforms(training_patch_size)

    model, trainer = train_regressor(
         folder_path=folder_path, 
         model_path=model_path, 
         batch_size=batch_size, # 64 seems to bring unstability
         learning_rate=learning_rate, #5e-3 leads to 0.489
         epochs=epochs,
         encoder=encoder, 
         transform_collection=transforms,
         in_chans=in_chans,
         test_size=test_size
         )
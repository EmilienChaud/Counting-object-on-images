from .utils import *



def get_a_random_training_sub_image(window, blobs):

    x_min = np.min(blobs[:,0])
    x_max = np.max(blobs[:,0])

    y_min = np.min(blobs[:,1])
    y_max = np.max(blobs[:,1])

    if x_min+int(window/2)> x_max-int(window/2):
        x = int((x_min+x_max)/2)
        if y_min+int(window/2)> y_max-int(window/2):
            print("this is a shit image")

    else:
        x = np.random.randint(x_min+int(window/2), x_max-int(window/2))

    if y_min+int(window/2)> y_max-int(window/2):
        x = int((y_min+y_max)/2)

    else:
        y = np.random.randint(y_min+int(window/2), y_max-int(window/2))

    mask = (blobs[:,0]>(x-window/2))*(blobs[:,0]<(x+window/2))*(blobs[:,1]>(y-window/2))*(blobs[:,1]<((y+window/2)))

    while sum(mask) == 0:
        x = np.random.randint(x_min+int(window/2), x_max-int(window/2))
        y = np.random.randint(y_min+int(window/2), y_max-int(window/2))
        mask = (blobs[:,0]>(x-window/2))*(blobs[:,0]<(x+window/2))*(blobs[:,1]>(y-window/2))*(blobs[:,1]<((y+window/2)))


    return x,y


    

def get_HM_mask(window, x,y, blobs, color_blobs):
    
    HM = np.zeros([window,window])
    mask = (blobs[:,0]>(x-window/2))*(blobs[:,0]<(x+window/2))*(blobs[:,1]>(y-window/2))*(blobs[:,1]<((y+window/2)))

    if sum(1*mask)==0:
        print(x, y)

    kind_blob = give_color_kind(color_blobs[mask])

    for k in range(blobs[mask].shape[0]):
        x_blob, y_blob, s = blobs[mask][k]
        if kind_blob[k] == 0:
            HM = HM + make_a_disk([window,window],
                                  int(x_blob-x+window/2),
                                  int(y_blob-y+window/2),
                                  60)
            #HM[int(x_blob-x+window/2),int(y_blob-y+window/2)]=50
        if kind_blob[k] == 1:
            HM = HM + make_a_disk([window,window],
                                  int(x_blob-x+window/2),
                                  int(y_blob-y+window/2),
                                  10)
            #HM[int(x_blob-x+window/2),int(y_blob-y+window/2)]=10
        if kind_blob[k] == 2:
            HM = HM + make_a_disk([window,window],
                                  int(x_blob-x+window/2),
                                  int(y_blob-y+window/2),
                                  60)
            #HM[int(x_blob-x+window/2),int(y_blob-y+window/2)]=50
        if kind_blob[k] == 3:
            HM = HM + make_a_disk([window,window],
                                  int(x_blob-x+window/2),
                                  int(y_blob-y+window/2),
                                  100)

            #HM[int(x_blob-x+window/2),int(y_blob-y+window/2)]=100
    return HM




def Get_training_sample_of_sub_images_from_on_image(path_data, filename, window,
                                                    nb_set):
    
    image_set = np.zeros([nb_set, window, window, 3])
    HM_set = np.zeros([nb_set, window, window, 1])
    
    image_dotted, image_normal = load_image(path_data, filename)
    
    blobs = get_point_position(image_dotted, image_normal)
    color_blobs = give_blobs_color(image_dotted, blobs)
    
    for k in range(nb_set):
        
        x, y = get_a_random_training_sub_image(window, blobs)
        image_set[k,:,:,:] = image_normal[int(x-window/2):int(x+window/2),
                                          int(y-window/2):int(y+window/2),:]
        HM_set[k,:,:,0] = get_HM_mask(window, x,y, blobs, color_blobs)
       
    
    return image_set, HM_set




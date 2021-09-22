# Voice-ML


IoT devices are becoming every day a more intrinsic part of our daily lives, one of the most natural and common ways to interface with these devices is through voice commands. With Voice ID we can take advantage of voice interfacing and add a variety of features for both security and convenience.

Voice ID uses an AI model to recognize a user by using his voice independently of what he is saying. This way a user can save his voice ID which can then be used by devices for multiple applications such as parental control, device locking or delivering specific content to a user.

Voice ID runs on the mobile device itself so it doesn’t require internet access, it can connect directly with other devices using a wifi or Bluetooth connection.  Since Voice ID will be optimized for hardware acceleration it could run on more specific devices other than a smartphone, allowing users to have more flexibility with their Voice ID’s


Model Training folder contains code for processing and training of tensorflow model

App folder contains android app that uses converted tensorflow model for verification

## Results

| Model       | EER          | 
| ------------- |:-------------:| 
| GMM -UBM      | 17.1 | 
|I-Vectors       | 12.8      |  
| I-Vectors + PLDA | 11.5      |
| CNN-20148 | 11.3      |
| CNN-256 + Pair Selections | 10.5      |
| Mobilenet+ Siamese (this repository) | 11.4      |

The Method used is Mobilenet+ Siamese, the other metrics are existing methods using the same dataset added for comparison, taken from Hossein Salehgaffaripaper “Speaker Verification using Convolutional NeuralNetworks”

## Performance

| Hardware Acceleration      | Enrollment(ms)        | Verification(ms) |
| ------------- |:-------------:| :------------:|
| No      | 642| 634| 
|Yes      | 171   | 191|

Tested on : Pixel 3 with Android 9

Using NNAPI for hardware acceleration reduces inference time by around 70%

## More Info

Report: https://1drv.ms/b/s!AvrId-VjUEW7g_RhsiMxPiswnVlRPQ?e=iopMA7

Report Chinese Version: https://1drv.ms/p/s!AvrId-VjUEW7hOR1v4vlDUSaknKN3Q?e=qy2Ey6 
        
App demo (Chinese): https://1drv.ms/v/s!AvrId-VjUEW7hOR2T1cYDO3X4g6Oow?e=3hMmjX 

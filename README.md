# Voice-ML

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

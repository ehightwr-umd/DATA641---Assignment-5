# Hypothesis Testing for Victims

## Cluster 2:
Observed counts:
cluster  False  True 
journal              
CNN         68     15
FOX         64     14
NYT         81     16
WSJ         52     11
Expected counts if null hypothesis holds:
cluster      False      True 
journal                      
CNN      68.520249  14.479751
FOX      64.392523  13.607477
NYT      80.077882  16.922118
WSJ      52.009346  10.990654
Deviation from expected (%):
cluster  False  True 
journal              
CNN       -0.8    3.6
FOX       -0.6    2.9
NYT        1.2   -5.4
WSJ       -0.0    0.1

Chi2=0.10, p-value=0.9922
→ Fail to reject null hypothesis: distribution similar across outlets

## Cluster 1:
Observed counts:
cluster  False  True 
journal              
CNN         68     15
FOX         69      9
NYT         84     13
WSJ         52     11
Expected counts if null hypothesis holds:
cluster      False      True 
journal                      
CNN      70.588785  12.411215
FOX      66.336449  11.663551
NYT      82.495327  14.504673
WSJ      53.579439   9.420561
Deviation from expected (%):
cluster  False  True 
journal              
CNN       -3.7   20.9
FOX        4.0  -22.8
NYT        1.8  -10.4
WSJ       -2.9   16.8

Chi2=1.85, p-value=0.6052
→ Fail to reject null hypothesis: distribution similar across outlets

## Cluster 6:
Observed counts:
cluster  False  True 
journal              
CNN         80      3
FOX         72      6
NYT         89      8
WSJ         52     11
Expected counts if null hypothesis holds:
cluster      False     True 
journal                     
CNN      75.760125  7.239875
FOX      71.196262  6.803738
NYT      88.538941  8.461059
WSJ      57.504673  5.495327
Deviation from expected (%):
cluster  False  True 
journal              
CNN        5.6  -58.6
FOX        1.1  -11.8
NYT        0.5   -5.4
WSJ       -9.6  100.2

Chi2=8.89, p-value=0.0308
→ Reject null hypothesis: distribution differs across outlets

# Hypothesis Testing for Shooters

## Cluster 1:
Observed counts:
cluster  False  True 
journal              
CNN         14      5
FOX         12      7
NYT          4      2
WSJ          8      7
Expected counts if null hypothesis holds:
cluster      False     True 
journal                     
CNN      12.237288  6.762712
FOX      12.237288  6.762712
NYT       3.864407  2.135593
WSJ       9.661017  5.338983
Deviation from expected (%):
cluster  False  True 
journal              
CNN       14.4  -26.1
FOX       -1.9    3.5
NYT        3.5   -6.3
WSJ      -17.2   31.1

Chi2=1.54, p-value=0.6726
→ Fail to reject null hypothesis: distribution similar across outlets

## Cluster 2:
Observed counts:
cluster  False  True 
journal              
CNN         12      7
FOX         16      3
NYT          3      3
WSJ          8      7
Expected counts if null hypothesis holds:
cluster      False     True 
journal                     
CNN      12.559322  6.440678
FOX      12.559322  6.440678
NYT       3.966102  2.033898
WSJ       9.915254  5.084746
Deviation from expected (%):
cluster  False  True 
journal              
CNN       -4.5    8.7
FOX       27.4  -53.4
NYT      -24.4   47.5
WSJ      -19.3   37.7

Chi2=4.64, p-value=0.2002
→ Fail to reject null hypothesis: distribution similar across outlets

## Cluster 3:
Observed counts:
cluster  False  True 
journal              
CNN         14      5
FOX         11      8
NYT          6      0
WSJ         14      1
Expected counts if null hypothesis holds:
cluster      False     True 
journal                     
CNN      14.491525  4.508475
FOX      14.491525  4.508475
NYT       4.576271  1.423729
WSJ      11.440678  3.559322
Deviation from expected (%):
cluster  False  True 
journal              
CNN       -3.4   10.9
FOX      -24.1   77.4
NYT       31.1 -100.0
WSJ       22.4  -71.9

Chi2=7.89, p-value=0.0482
→ Reject null hypothesis: distribution differs across outlets

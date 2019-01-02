# steamGameRec
Recommendation engine for Steam games based on data collected from Steam Web API. Please see the attached report for complete details of the project

## Data
Userid-playtime data is collected from the Steam Web API. Since Steam does not maintain explicit ratings for games, we derive our ratings based on playtime.

## Models
Two classes of models were developed:
- Binary Ratings (Recommendations based on whether user bought the game or not)
- Continuous Ratings (Recommendations based on continuous ratings on the [0,1] scale)

## Continous Ratings Models
Pleaes refer to the notebook ContinousRatingSurprise.ipynb for continous model. This has been implemented using the surprise library.

## Results
### Binary Models
| Model               | Precision@10 | Recall@10 |
|---------------------|--------------|-----------|
| Baseline 1          | 0.0924       | 0.0352    |
| Baseline 2          | 0.1086       | 0.0468    |
| Pearson Correlation | 0.4625       | 0.2057    |
| Cosine Similarity   | 0.4746       | 0.2089    |
| Implicit Linear     | 0.5012       | 0.2287    |
| Implicit Log        | 0.5580       | 0.2559    |


### Continuous Models
| Model                | RMSE   |
|----------------------|--------|
| Baseline             | 0.2871 |
| Cosine Similarity    | 0.2660 |
| Matrix Factorization | 0.2526 |

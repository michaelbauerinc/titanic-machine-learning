# titanic-machine-learning
Simple machine learning POC that uses linear regression to estimate survival chance on the titanic crash.

The datasets used to train this model can be found [here](https://storage.googleapis.com/tf-datasets/titanic/train.csv) for the training, and [here](https://storage.googleapis.com/tf-datasets/titanic/eval.csv) for the evaluation data.

## Dependencies

- Docker desktop or your favorite container platform

## How to Use

To start the app, run the following on a unix system

```
./start.sh
```

To start the app in local-dev with a volume to your local machine, run the following on a unix system:

```
./start-dev.sh
```


After the model compiles, the user will be prompted for the following values:

sex (string, i.e male or female)
class (string, i.e First, Second, Third)
deck (string, i.e A, B, C, D, E, F, G, unknown)
embark_town (string, i.e Cherbourg, Queenstown)
alone (string, i.e y, n)

age (float, i.e 22.0, 54.55)
fare (float, i.e 5.50, 45.00)


n_siblings_spouses (int, i.e 0, 1)
parch (int, i.e 1, 4, 5)

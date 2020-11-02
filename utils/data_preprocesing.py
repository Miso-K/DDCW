import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from arff2pandas import a2p

def read_elec_norm_data(filename):
    """
    Function read elec_norm dataset and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    dataset = pd.read_csv(filename)

    X = pd.DataFrame(dataset.iloc[:, 1:-1])
    y = pd.DataFrame(dataset.iloc[:, -1])

    scaler = MinMaxScaler()

    X = pd.DataFrame(scaler.fit_transform(X))

    label = np.unique(y)
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)
    y = pd.DataFrame(y)
    y.astype('int32')

    data = pd.concat([X, y], axis=1)
    data = data.values

    return data, X, y

def read_kdd_data_multilable(filename):
    """
    Function read kdd kdd multilable data and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                 "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
                 "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                 "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "label"]

    data_10percent = pd.read_csv(filename, names=col_names, header=None)

    X = pd.DataFrame(data_10percent.iloc[:, :-1])
    y = pd.DataFrame(data_10percent.iloc[:, -1])

    le = LabelEncoder()
    for col in X.columns.values:
        if X[ col ].dtypes == 'object':
            le.fit(X[ col ])
            X[ col ] = le.transform(X[ col ])

    y = pd.DataFrame(le.fit_transform(y))
    y.astype('int32')

    scaler = MinMaxScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X))
    data = pd.concat([ X_sc, y ], axis=1)
    data = data.values

    return data, X_sc, y


def read_data_arff(filename):
    """
    Function read arff data and prepare X and y data
    :param filename:
    :return: data, X, y
    """
    with open(filename) as f:
        df = a2p.load(f)

    X = pd.DataFrame(df.iloc[:, :-1])
    y = pd.DataFrame(df.iloc[:, -1])

    le = LabelEncoder()
    for col in X.columns.values:
        if X[ col ].dtypes == 'object':
            le.fit(X[ col ])
            X[ col ] = le.transform(X[ col ])

    y = pd.DataFrame(le.fit_transform(y))
    y.astype('int32')

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    data = pd.concat([ X, y ], axis=1)
    data = data.values

    return data, X, y

def read_data_csv(filename):
    """
    Function read csv data and filter X and y data
    :param filename:
    :return: data, X, y
    """
    df = pd.read_csv(filename)

    X = pd.DataFrame(df.iloc[:, :-1])
    y = pd.DataFrame(df.iloc[:, -1])
    data = []
    data = pd.concat([ X, y ], axis=1)
    data = data.values

    return data, X, y
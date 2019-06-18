import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.utils import shuffle
from scipy.io import arff
import chartcolumn
import fileinput
from sklearn.model_selection import train_test_split
from sklearn import svm
import glob as gl
seed = 0

"Normalize training and testing sets"
def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X


"*************************Read Shuttle data*****************************"
def ctu13(datasetname,pathTrain, pathTest, PathColumn):
    "*************************Chosing dataset*********************************"
    d = np.genfromtxt(pathTrain, delimiter=",")
    d = d[~np.isnan(d).any(axis=1)]    #discard the '?' values

    np.random.seed(seed)
    np.random.shuffle(d)

    dX = d[:,0:-1]              #put data to dX without the last column (labels)
    dy = d[:,-1] 
    
    #print("Normal: %d Anomaly %d" %(len(dX0), len(dX1)))
    split = 0.4
   
    idx0  = int(split * len(dX))
    

    X_train = dX[:idx0]        # train_X is 80% of the normal 
    Y_train = dy [:idx0]
    X_test = dX[idx0:]
    Y_test = dy[idx0:]
    X_train, X_test = normalize_data(X_train, X_test,'minmax')
    X0 = X_train[(Y_train==1)]
    X1 = X_train[(Y_train!=1)]
    Y_train = pd.get_dummies(Y_train)
    Y_test = pd.get_dummies(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
   
    #X_train, X_test = normalize_data(X_train, X_test,'minmax')
    return X_train, Y_train, X_test, Y_test, X0, X1


def unsw(pathTrain, pathTest, pathColumn):

    features = ["dur","proto","service","state","spkts","dpkts","sbytes",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "dbytes","rate","sttl","dttl","sload","dload","sloss","dloss",
                "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin",
                "tcprtt","synack","ackdat","smean","dmean","trans_depth",
                "response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm",
                "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login",
                 "ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports"]
    labels = ["label"]
    label9 = ["attack_cat"]

    

    TrainData =pd.read_csv(pathTrain,sep=',')

    TestData = pd.read_csv(pathTest,sep=',')
    X_train = TrainData[features]
    X_test = TestData[features]


    Y_train2 = TrainData[labels]
    Y_train9 = TrainData[label9]
    Y9_tr = TrainData[label9]

    Y_test2 = TestData[labels]
    Y_test9 = TestData[label9]
    Y9_te = TestData[label9]


#process label
    Y_train = pd.get_dummies(Y_train2,  columns = labels)
    Y_test = pd.get_dummies(Y_test2,  columns = labels)

    Y_train9 = pd.get_dummies(Y_train9,  columns = label9)
    Y_test9 = pd.get_dummies(Y_test9,  columns = label9)

 #process with X
    l = len(X_train)
#encode
    data=X_train.append(X_test)
    data = pd.get_dummies(data, columns = ["proto","service","state"])

    X_train = data [0:l]
    X_test = data [l:len(data)]
    
    
    #convert to array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train2 = np.array(Y_train2)
    Y_test2 = np.array(Y_test2)
    Y_train01  = np.asarray(Y_train2)
    print ("shape Y ",Y_train01.shape)
    Y_train01 = np.reshape(Y_train01, (len(Y_train01)))
    X0 = X_train[Y_train01 == 0]
    X1 = X_train[Y_train01 != 0]
    X_train, X_test = normalize_data(X_train, X_test,'minmax')

    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_train9 = np.array(Y_train9)
    Y_test9 = np.array(Y_test9)

    Y9_tr = np.array(Y9_tr)
    Y9_te = np.array(Y9_te)

    unique1, counts1 = np.unique(Y9_tr, return_counts=True)
    print (dict(zip(unique1, counts1)))
    unique2, counts2 = np.unique(Y9_te, return_counts=True)
    print (dict(zip(unique2, counts2)))
    X0 = np.array(X0)
    X1 = np.array(X1)
    return  X_train, Y_train, X_test, Y_test, X0, X1

#-------------------------------------------------------------------------------------

def nslkdd(pathTrain, pathTest, PathColumn):

    pathTrain = "nslkdd/KDDTrain+.txt"
    pathTest = "nslkdd/KDDTest+.txt"

    col_names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]
    features = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","labels"]
    col_labels = ["dst_host_srv_rerror_rate"]



    TrainData =pd.read_csv(pathTrain, header=None, names = col_names)
    print (TrainData.dtypes)
    TestData = pd.read_csv(pathTest, header=None, names = col_names)
    X_train = TrainData[features]
    X_test = TestData[features]
    
    
    #for 5 classes
    Y_train = TrainData[col_labels]
    Y_test = TestData[col_labels]
    
    #For two classes
    Y_train2 = Y_train
    Y_test2 = Y_test
     #two classes
    Ytr = Y_train2 == 'normal'
    Yte = Y_test2 == 'normal'
    
    Y_train2 = pd.get_dummies(Ytr, columns = ["dst_host_srv_rerror_rate"])
    Y_test2 = pd.get_dummies(Yte,  columns = ["dst_host_srv_rerror_rate"])
    
    Ytr = Ytr + 0
    Yte = Yte + 0
    #Preprocess with label
    #normal = 0
    #dos = 1
    #u2r = 2
    #r2l = 3
    #probe = 4
    Y_train[Y_train == 'normal'] = 0#'dos'
    Y_train[Y_train == 'back'] = 1#'dos'
    Y_train[Y_train == 'buffer_overflow'] = 2# 'u2r'
    Y_train[Y_train == 'ftp_write'] = 3# 'r2l'
    Y_train[Y_train == 'guess_passwd'] = 3#'r2l'
    Y_train[Y_train == 'imap'] = 3#'r2l'
    Y_train[Y_train == 'ipsweep'] = 4#'probe'
    Y_train[Y_train == 'land'] = 1#'dos' 
    Y_train[Y_train == 'loadmodule'] = 2#'u2r'
    Y_train[Y_train == 'multihop'] = 3#'r2l'
    Y_train[Y_train == 'neptune'] = 1#'dos'
    Y_train[Y_train == 'nmap'] = 4#'probe'
    Y_train[Y_train == 'perl'] = 2#'u2r'
    Y_train[Y_train == 'phf'] =  3#'r2l'
    Y_train[Y_train == 'pod'] =  1#'dos'
    Y_train[Y_train == 'portsweep'] = 4#'probe'
    Y_train[Y_train == 'rootkit'] = 2#'u2r'
    Y_train[Y_train == 'satan'] = 4#'probe'
    Y_train[Y_train == 'smurf'] = 1#'dos'
    Y_train[Y_train == 'spy'] = 3#'r2l'
    Y_train[Y_train == 'teardrop'] = 1#'dos'
    Y_train[Y_train == 'warezclient'] = 3#'r2l'
    Y_train[Y_train == 'warezmaster'] = 3#'r2l'
    Y_train[Y_train == 'mailbomb'] = 1#'dos'
    Y_train[Y_train == 'processtable'] = 1#'dos'
    Y_train[Y_train == 'udpstorm'] = 1#'dos'
    Y_train[Y_train == 'apache2'] = 1#'dos'
    Y_train[Y_train == 'worm'] = 1#'dos'

    Y_train[Y_train == 'xlock'] = 3#'r2l'
    Y_train[Y_train == 'xsnoop'] = 3#'r2l'	
    Y_train[Y_train == 'snmpguess'] = 3#'r2l'
    Y_train[Y_train == 'snmpgetattack'] = 3#'r2l'
    Y_train[Y_train == 'httptunnel'] = 3#'r2l'
    Y_train[Y_train == 'sendmail'] = 3#'r2l'	
    Y_train[Y_train == 'named'] = 3#'r2l'	

    Y_train[Y_train == 'sqlattack'] = 2#'u2r'
    Y_train[Y_train == 'xterm'] = 2#'u2r'
    Y_train[Y_train == 'ps'] = 2#'u2r'

    Y_train[Y_train == 'mscan'] = 4#'probe'
    Y_train[Y_train == 'saint'] = 4#'probe'
    
    #testing label
    Y_test[Y_test == 'normal'] = 0#'dos'
    Y_test[Y_test == 'back'] = 1#'dos'
    Y_test[Y_test == 'buffer_overflow'] = 2# 'u2r'
    Y_test[Y_test == 'ftp_write'] = 3# 'r2l'
    Y_test[Y_test == 'guess_passwd'] = 3#'r2l'
    Y_test[Y_test == 'imap'] = 3#'r2l'
    Y_test[Y_test == 'ipsweep'] = 4#'probe'
    Y_test[Y_test == 'land'] = 1#'dos' 
    Y_test[Y_test == 'loadmodule'] = 2#'u2r'
    Y_test[Y_test == 'multihop'] = 3#'r2l'
    Y_test[Y_test == 'neptune'] = 1#'dos'
    Y_test[Y_test == 'nmap'] = 4#'probe'
    Y_test[Y_test == 'perl'] = 2#'u2r'
    Y_test[Y_test == 'phf'] =  3#'r2l'
    Y_test[Y_test == 'pod'] =  1#'dos'
    Y_test[Y_test == 'portsweep'] = 4#'probe'
    Y_test[Y_test == 'rootkit'] = 2#'u2r'
    Y_test[Y_test == 'satan'] = 4#'probe'
    Y_test[Y_test == 'smurf'] = 1#'dos'
    Y_test[Y_test == 'spy'] = 3#'r2l'
    Y_test[Y_test == 'teardrop'] = 1#'dos'
    Y_test[Y_test == 'warezclient'] = 3#'r2l'
    Y_test[Y_test == 'warezmaster'] = 3#'r2l'
    Y_test[Y_test == 'mailbomb'] = 1#'dos'
    Y_test[Y_test == 'processtable'] = 1#'dos'
    Y_test[Y_test == 'udpstorm'] = 1#'dos'
    Y_test[Y_test == 'apache2'] = 1#'dos'
    Y_test[Y_test == 'worm'] = 1#'dos'

    Y_test[Y_test == 'xlock'] = 3#'r2l'
    Y_test[Y_test == 'xsnoop'] = 3#'r2l'	
    Y_test[Y_test == 'snmpguess'] = 3#'r2l'
    Y_test[Y_test == 'snmpgetattack'] = 3#'r2l'
    Y_test[Y_test == 'httptunnel'] = 3#'r2l'
    Y_test[Y_test == 'sendmail'] = 3#'r2l'	
    Y_test[Y_test == 'named'] = 3#'r2l'	

    Y_test[Y_test == 'sqlattack'] = 2#'u2r'
    Y_test[Y_test == 'xterm'] = 2#'u2r'
    Y_test[Y_test == 'ps'] = 2#'u2r'

    Y_test[Y_test == 'mscan'] = 4#'probe'
    Y_test[Y_test == 'saint'] = 4#'probe'
    
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    Y_train5 = pd.get_dummies(Y_train,  columns = ["dst_host_srv_rerror_rate"])
    Y_test5 = pd.get_dummies(Y_test,  columns = ["dst_host_srv_rerror_rate"])
    #print (Y_train2[0:10])
    #print (Ytr[0:10])
    
    #process with X
     #SCALING
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #one hot features
    l = len(X_train)
    data=X_train.append(X_test)
    data = pd.get_dummies(data, columns = ['duration', 'protocol_type', 'service'])
    #scale
    #data = pandas.DataFrame(scaler.fit_transform(data), columns=data.columns)

    X_train = data [0:l]
    X_test = data [l:len(data)]
    
   
    
    #from sklearn.preprocessing import LabelEncoder
    #le=LabelEncoder()
    # Iterating over all the common columns in train and test
    #for col in X_test.columns.values:
       # Encoding only categorical variables
    #   if X_test[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
    #        data=X_train[col].append(X_test[col])
            
    #        le.fit(data.values)
    #        X_train[col]=le.transform(X_train[col])
    #        X_test[col]=le.transform(X_test[col])

    

    #X_train = pandas.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    #X_test = pandas.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

    #convert to array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train, X_test = normalize_data(X_train, X_test,'minmax')
    
    Y_train2 = np.array(Y_train2)
    Y_test2 = np.array(Y_test2)
    
    Y_train5 = np.array(Y_train5)
    Y_test5 = np.array(Y_test5)
    
    print (Y_train5[0:10])
    Y5 = np.argmax(Y_train5, 1)

    Y5 = np.array(Y5)
    print (Y5)
    
    X0 = X_train[Y5 == 0]
    X1 = X_train[Y5 != 0]
    X2 = X_train[Y5 == 2]
    X3 = X_train[Y5 == 3]
    X4 = X_train[Y5 == 4]
  
    X0 = np.array(X0)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
   
    
    
    unique1, counts1 = np.unique(Y_train, return_counts=True)
    print (dict(zip(unique1, counts1)))
    unique2, counts2 = np.unique(Y_test, return_counts=True)
    print (dict(zip(unique2, counts2)))
    #return  X_train, Y_train2, X_test, Y_test2, Y_train5, Y_test5, X0, X1, X2, X3, X4
    return  X_train, Y_train2, X_test, Y_test2, X0, X1

#------------------------------------------------------------------------------------

def Spam(pathTrain, pathTest, PathColumn):
    col_names = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
            "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
            "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
            "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
            "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl",
            "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
            "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts",
            "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
            "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
            "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
            "capital_run_length_total", "class"]
    
    readfileSpamBase = np.genfromtxt(pathTrain, delimiter=",", dtype=np.float32, names=col_names)
    np.random.shuffle(readfileSpamBase)
    spambaseData = pd.DataFrame(readfileSpamBase)
    print(spambaseData)

    spambaseDataTrain = spambaseData[:3681]
    spambaseDataTest = spambaseData[3681:]

    Y_Train = spambaseDataTrain['class']
    Y_Train = list(np.array(Y_Train, dtype=np.int32))
    Y_Train = pd.get_dummies(Y_Train)
    X_Train = spambaseDataTrain.drop("class", axis=1)
    
    Y_Test = spambaseDataTest['class']
    Y_Test = list(np.array(Y_Test, dtype=np.int32))
    Y_Test = pd.get_dummies(Y_Test)
    X_Test = spambaseDataTest.drop('class', axis=1)
    X_Train, X_Test = normalize_data(X_Train, X_Test, "minmax")

    X_0 = X_Train[spambaseDataTrain["class"] == 0]
    X_1 = X_Train[spambaseDataTrain["class"] == 1]
    
    X_Train = np.array (X_Train)
    X_Test = np.array (X_Test)
    Y_Train = np.array (Y_Train)
    Y_Test = np.array (Y_Test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)

    return X_Train,Y_Train, X_Test, Y_Test, X_0, X_1

#*******************************************Phishing*********************************
def Phishing(pathTrain, pathTest, pathColum):
    phishingWebsiteDatasets, meta = arff.loadarff(open(pathTrain, 'r'))
    np.random.shuffle(phishingWebsiteDatasets)

    phishingWebsiteDatasets = pd.DataFrame(phishingWebsiteDatasets)
    phishingWebsiteDatasets["having_IP_Address"] = phishingWebsiteDatasets["having_IP_Address"].str.decode('utf-8')
    phishingWebsiteDatasets["URL_Length"] = phishingWebsiteDatasets["URL_Length"].str.decode('utf-8')
    phishingWebsiteDatasets["Shortining_Service"] = phishingWebsiteDatasets["Shortining_Service"].str.decode('utf-8')
    phishingWebsiteDatasets["having_At_Symbol"] = phishingWebsiteDatasets["having_At_Symbol"].str.decode('utf-8')
    phishingWebsiteDatasets["double_slash_redirecting"] = phishingWebsiteDatasets["double_slash_redirecting"].str.decode('utf-8')
    phishingWebsiteDatasets["Prefix_Suffix"] = phishingWebsiteDatasets["Prefix_Suffix"].str.decode('utf-8')
    phishingWebsiteDatasets["having_Sub_Domain"] = phishingWebsiteDatasets["having_Sub_Domain"].str.decode('utf-8')
    phishingWebsiteDatasets["SSLfinal_State"] = phishingWebsiteDatasets["SSLfinal_State"].str.decode('utf-8')
    phishingWebsiteDatasets["Domain_registeration_length"] = phishingWebsiteDatasets["Domain_registeration_length"].str.decode('utf-8')
    phishingWebsiteDatasets["Favicon"] = phishingWebsiteDatasets["Favicon"].str.decode('utf-8')
    phishingWebsiteDatasets["port"] = phishingWebsiteDatasets["port"].str.decode('utf-8')
    phishingWebsiteDatasets["HTTPS_token"] = phishingWebsiteDatasets["HTTPS_token"].str.decode('utf-8')
    phishingWebsiteDatasets["Request_URL"] = phishingWebsiteDatasets["Request_URL"].str.decode('utf-8')
    phishingWebsiteDatasets["URL_of_Anchor"] = phishingWebsiteDatasets["URL_of_Anchor"].str.decode('utf-8')
    phishingWebsiteDatasets["Links_in_tags"] = phishingWebsiteDatasets["Links_in_tags"].str.decode('utf-8')
    phishingWebsiteDatasets["SFH"] = phishingWebsiteDatasets["SFH"].str.decode('utf-8')
    phishingWebsiteDatasets["Submitting_to_email"] = phishingWebsiteDatasets["Submitting_to_email"].str.decode('utf-8')
    phishingWebsiteDatasets["Abnormal_URL"] = phishingWebsiteDatasets["Abnormal_URL"].str.decode('utf-8')
    phishingWebsiteDatasets["Redirect"] = phishingWebsiteDatasets["Redirect"].str.decode('utf-8')
    phishingWebsiteDatasets["on_mouseover"] = phishingWebsiteDatasets["on_mouseover"].str.decode('utf-8')
    phishingWebsiteDatasets["RightClick"] = phishingWebsiteDatasets["RightClick"].str.decode('utf-8')
    phishingWebsiteDatasets["popUpWidnow"] = phishingWebsiteDatasets["popUpWidnow"].str.decode('utf-8')
    phishingWebsiteDatasets["Iframe"] = phishingWebsiteDatasets["Iframe"].str.decode('utf-8')
    phishingWebsiteDatasets["age_of_domain"] = phishingWebsiteDatasets["age_of_domain"].str.decode('utf-8')
    phishingWebsiteDatasets["DNSRecord"] = phishingWebsiteDatasets["DNSRecord"].str.decode('utf-8')
    phishingWebsiteDatasets["web_traffic"] = phishingWebsiteDatasets["web_traffic"].str.decode('utf-8')
    phishingWebsiteDatasets["Page_Rank"] = phishingWebsiteDatasets["Page_Rank"].str.decode('utf-8')
    phishingWebsiteDatasets["Google_Index"] = phishingWebsiteDatasets["Google_Index"].str.decode('utf-8')
    phishingWebsiteDatasets["Links_pointing_to_page"] = phishingWebsiteDatasets["Links_pointing_to_page"].str.decode('utf-8')
    phishingWebsiteDatasets["Statistical_report"] = phishingWebsiteDatasets["Statistical_report"].str.decode('utf-8')
    phishingWebsiteDatasets["Result"] = phishingWebsiteDatasets["Result"].str.decode('utf-8')
    
    phishingWebsiteDatasets_Train = phishingWebsiteDatasets[:8844]
    phishingWebsiteDatasets_Test = phishingWebsiteDatasets[8844:]
    
    Y_Train = np.array(phishingWebsiteDatasets_Train["Result"])
    Y_Train[Y_Train == '1'] = '0'
    Y_Train[Y_Train == '-1'] = '1'
    
    Y_Train = pd.get_dummies(Y_Train)
    
    
    Y_Test = np.array(phishingWebsiteDatasets_Test["Result"])
    Y_Test[Y_Test == '1'] = '0'
    Y_Test[Y_Test == '-1'] = '1'
    
    Y_Test = pd.get_dummies(Y_Test)
   
    
    X_Test = phishingWebsiteDatasets_Test.drop("Result", axis=1)
    X_Train = phishingWebsiteDatasets_Train.drop("Result", axis=1)
    
    X_Test = pd.get_dummies(X_Test)
    X_Train = pd.get_dummies(X_Train)
    
    X_Train, X_Test = normalize_data(X_Train, X_Test, "minmax")
    
    X_1 = X_Train[phishingWebsiteDatasets_Train["Result"] == '-1']
    X_0 = X_Train[phishingWebsiteDatasets_Train["Result"] == '1']
    
    X_Train = np.array(X_Train, dtype=np.float64)
    Y_Train = np.array(Y_Train, dtype=np.float64)
    X_Test = np.array(X_Test, dtype=np.float64)
    Y_Test = np.array(Y_Test, dtype=np.float64)
    X_0 = np.array (X_0, dtype=np.float64)
    X_1 = np.array (X_1, dtype=np.float64)
    
    return X_Train, Y_Train, X_Test, Y_Test, X_0, X_1

#******************************************VirusShare********************************
def take_index(arr):
    array_index = []
    array_values = []
    array_X = []
    for index in range(1, len(arr), 2):
        array_index.append(arr[index])
    #print(array_index)

    for val in range(2, len(arr), 2):
        array_values.append(arr[val])
    #print(array_values)

    for i in range(0, 479):
        if str(i) in array_index:
            array_X.append(int(array_values[0]))
            del array_values[0]
        else:
            array_X.append(0)
    return array_X
def VirusShare(pathTrain, pathTest, PathColumn):
    path = 'VirusShare'
    all_files = gl.glob(path+'/*.txt')

    X = []
    Y = []

    dataframe = pd.DataFrame()
    for filename in all_files:
        data = open(filename).readlines()

        np.random.shuffle(data)
        for i in data:
            i = i.replace(':', " ").split(' ')
            Y.append(float(i[0]))
            array_X = take_index(i)
            X.append(np.array(array_X))
    Y = np.array(Y)
    Y[Y < 0.5] = 0
    Y[Y > 0.5] = 1
    dataframe = dataframe.append(pd.DataFrame(X), sort=True)
    dataframe['class'] = np.array(Y)
    print(dataframe)

    dataframe_train = dataframe[:86285]
    dataframe_test = dataframe[86285:]

    Y_train = dataframe_train['class']
    Y_train = pd.get_dummies(Y_train)

    Y_test = dataframe_test['class']
    Y_test = pd.get_dummies(Y_test)

    X_train = dataframe_train.drop('class', axis=1)
    X_test = dataframe_test.drop('class', axis=1)

    #X_train, X_test = normalize_data(X_train, X_test, 'minmax')

    X_0 = X_train[dataframe_train['class'] == 0]
    X_1 = X_train[dataframe_train['class'] == 1]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
   
    return X_train, Y_train, X_test, Y_test, X_0, X_1
#******************************************Antivirus********************************
def take_X(arr):
    array_X = []
    for i in range(1, 532):
        if str(i) in arr:
            array_X.append(1)
        else:
            array_X.append(0)
    return array_X
def Antivirus(pathTrain, pathTest, PathColumn):
    data_train = open('Antivirus/dataset.train').readlines()
    data_test = open('Antivirus/Tst.test').readlines()
    dataframe = pd.DataFrame()
    Y = []
    X = []
    for i in data_train:
        i = i.replace(':', " ").split(' ')
        Y.append(i[0])
        array_X = take_X(i)
        X.append(np.array(array_X))

    Y = np.array(Y)
    Y[Y == '+1'] = 0
    Y[Y == '-1'] = 1
    dataframe = dataframe.append(pd.DataFrame(X))
    dataframe['class'] = np.array(Y, dtype=np.int32)
    dataframe = shuffle(dataframe)
    print(dataframe)
    dataframe_Train = dataframe[:298]
    dataframe_Test = dataframe[298:]

    X_train = dataframe_Train.drop('class', axis=1)
    Y_train = dataframe_Train['class']
    Y_train = pd.get_dummies(Y_train)

    X_test = dataframe_Test.drop('class', axis=1)
    Y_test = dataframe_Test['class']
    Y_test = pd.get_dummies(Y_test)
    X_train, X_test = normalize_data(X_train, X_test, 'minmax')

    X_0 = X_train[dataframe_Train['class'] == 0]
    X_1 = X_train[dataframe_Train['class'] == 1]
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)

    return X_train, Y_train, X_test, Y_test, X_0, X_1
#*******************************************Internet_ads****************************************
def handle_colname(name):
    col_name = []
    for i in name:
        if i[0] != "|":
            if ':' in i:
                col_name.append(i[0:i.index(':')])
    col_name.append('class')
    return col_name

def Internet_ads(pathTrain, pathTest, pathColum):
    column_names = handle_colname(open(pathColum).readlines())
    internet_ads = np.genfromtxt(pathTrain, delimiter=",", names=column_names)
    
    np.random.shuffle(internet_ads)
    internet_ads = pd.DataFrame(internet_ads)
    #internet_ads = internet_ads.dropna()

    internet_ads_Train = internet_ads[:2622]
    internet_ads_Test = internet_ads[2622:]

    
    Y_Test = list(np.array(internet_ads_Test['class'], dtype=np.int))
    Y_Test = pd.get_dummies(Y_Test)
    Y_Train = list(np.array(internet_ads_Train['class'], dtype=np.int))
    Y_Train = pd.get_dummies(Y_Train)

    X_Test = internet_ads_Test.drop('class', axis=1)
    X_Train = internet_ads_Train.drop('class', axis=1)
    
    X_Train, X_Test = normalize_data(X_Train, X_Test, "minmax")
    
    X_0 = X_Train[internet_ads_Train['class'] == 0]
    X_1 = X_Train[internet_ads_Train['class'] == 1]
    X_Test_0 = X_Test[internet_ads_Test['class'] == 0]
    X_Test_1 = X_Test[internet_ads_Test['class'] == 1]
    chartcolumn.chart_column(len(X_Train), len(X_Test), len(X_0), len(X_1), len(X_Test_0), len(X_Test_1), 'Ads')

    X_Train = np.array(X_Train)
    Y_Train = np.array(Y_Train)
    X_Test = np.array(X_Test)
    Y_Test = np.array(Y_Test)
    X_0 = np.array (X_0)
    X_1 = np.array (X_1)
    
    return X_Train, Y_Train, X_Test, Y_Test, X_0, X_1


#******************************************IoT********************************
def IoT_botnet(pathTrain, pathTest, pathColumn):
    dataframes_benign = pd.read_csv(pathTrain)[:30000]
    dataframes_nonbenign = pd.read_csv(pathTest)[:100000]

    dataframes_benign['class'] = 0
    dataframes_nonbenign['class'] = 1

    IoT_Botnet_data = dataframes_benign.append(dataframes_nonbenign, sort= True, ignore_index= True)
    print(IoT_Botnet_data)
    IoT_Botnet_data = shuffle(IoT_Botnet_data)

    IoT_Botnet_data_train = IoT_Botnet_data[0:int(len(IoT_Botnet_data)*70/100)]
    IoT_Botnet_data_test = IoT_Botnet_data[int(len(IoT_Botnet_data)*70/100):]

    print(IoT_Botnet_data_train)

    Y_train = IoT_Botnet_data_train['class']
    Y_train = pd.get_dummies(Y_train)

    Y_test = IoT_Botnet_data_test['class']
    Y_test = pd.get_dummies(Y_test)

    X_train = IoT_Botnet_data_train.drop("class", axis=1)
    X_test = IoT_Botnet_data_test.drop("class", axis=1)

    X_train, X_test = normalize_data(X_train, X_test, 'minmax')

    X_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    X_test_0 = X_train[IoT_Botnet_data_train["class"] == 0]
    X_test_1 = X_train[IoT_Botnet_data_train["class"] == 1]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
    return X_train, Y_train, X_test, Y_test, X_0, X_1


    
"*****************************Load dataset*****************************"
def read_data_sets(dataset,pathTrain, pathTest, PathColumn):
    path_data = "/home/cnm02/DATA/"
    NSLKDD = ["Probe", "DoS", "R2L", "U2R", "NSLKDD"]
    UNSW   = ["Fuzzers", "Analysis", "Backdoor", "DoS_UNSW", "Exploits", "Generic",\
            "Reconnaissance", "Shellcode", "Worms", "UNSW"]
    CTU13  = ["CTU13_06","CTU13_07","CTU13_08","CTU13_09","CTU13_10","CTU13_12","CTU13_13"]

    if (dataset == "unsw"):
        X_train, Y_train, X_test, Y_test, X0, X1 = unsw(pathTrain, pathTest, PathColumn)
    elif (dataset == "nslkdd"):
        X_train, Y_train, X_test, Y_test, X0, X1 = nslkdd(pathTrain, pathTest, PathColumn)
    elif (dataset == "ctu13_8"):
        X_train, Y_train, X_test, Y_test, X0, X1 = ctu13("ctu13_8",pathTrain, pathTest, PathColumn)#

    elif (dataset == "ctu13_10"):
        X_train, Y_train, X_test, Y_test, X0, X1 = ctu13("ctu13_10",pathTrain, pathTest, PathColumn)
    elif (dataset == "ctu13_13"):
        X_train, Y_train, X_test, Y_test, X0, X1 = ctu13("ctu13_13",pathTrain, pathTest, PathColumn)
    elif (dataset == "VirusShare"):
        X_train, Y_train, X_test, Y_test, X0, X1 = VirusShare(pathTrain, pathTest, PathColumn)
    elif (dataset == "Antivirus"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Antivirus(pathTrain, pathTest, PathColumn)
    elif (dataset == "IoT"):
        X_train, Y_train, X_test, Y_test, X0, X1 = IoT_botnet(pathTrain, pathTest, PathColumn)
    elif (dataset == "Phishing"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Phishing(pathTrain, pathTest, PathColumn)
    elif (dataset == "Spam"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Spam(pathTrain, pathTest, PathColumn)
    elif (dataset == "Ads"):
        X_train, Y_train, X_test, Y_test, X0, X1 = Internet_ads(pathTrain, pathTest, PathColumn)
    else:
        print ("Incorrect data")

    
   
    class DataSets(object):
          pass
    data_sets = DataSets()
    data_sets.train = DataSet(X_train, Y_train, X0, X1)
    data_sets.test = DataSet(X_test, Y_test, X0, X1)

    return data_sets

class DataSet(object):
  def __init__(self, features, labels,features0, features1, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert features.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (features.shape,
                                                 labels.shape))
      self._num_examples = features.shape[0]
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._features0 = features0
    self._features1 = features1
  @property
  def features(self):
    return self._features
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  @property
  def features0(self):
    return self._features0
  
  @property
  def features1(self):
    return self._features1
  
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(38)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._features[start:end], self._labels[start:end]

  def next_batch_100(self, batch_size):
        #major: 0
        #minor: 1
        X_ma = self.features0
        X_mi = self.features1
        
        size_class = len (X_mi)
        class_num = 2
        
        idx = np.random.randint(len(X_ma), size=size_class)
        X0 = X_ma[idx]
        Y0 = np.zeros((size_class, 1))

        idx = np.random.randint(len(X_mi), size=size_class)
        X1 = X_mi[idx]
        #eps = np.random.uniform(-0.001,0.001)
        #X1 = X1 + eps
        Y1 = np.ones((size_class, 1))

        self._features = np.concatenate((X0, X1),axis = 0)
        Y = np.concatenate((Y0, Y1),axis = 0)
        
        #convert to one-hot vector
        Y = (np.arange(class_num) ==Y[:,None]).astype(np.float32)
        Y =Y[:,0,:]
        Y = Y.astype(int)
        
        self._labels = Y

        self._num_examples = len (self._features)
        
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self._features[perm]
        self._labels = self._labels[perm]
        
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._features = self._features[perm]
          #self._features = self._features + eps
          self._labels = self._labels[perm]
          #self._cost = self._cost[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]#, self._cost[start:en
    
 
def next_batch_dss(self, batch_size, X_ma, X_mi):
        #major: 0
        #minor: 1
        size_class = len (X_mi)
        class_num = 2

        idx = np.random.randint(len(X_ma), size=size_class)
        X0 = X_ma[idx]
        Y0 = np.zeros((size_class, 1))

        idx = np.random.randint(len(X_mi), size=size_class)
        X1 = X_mi[idx]
        eps = np.random.uniform(-0.001,0.001)
        X1 = X1 + eps
        Y1 = np.ones((size_class, 1))

        self._features = np.concatenate((X0, X1),axis = 0)
        Y = np.concatenate((Y0, Y1),axis = 0)
        
        #convert to one-hot vector
        Y = (np.arange(class_num) ==Y[:,None]).astype(np.float32)
        Y =Y[:,0,:]
        Y = Y.astype(int)
        
        self._labels = Y

        self._num_examples = len (self._features)
        
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self._features[perm]
        self._labels = self._labels[perm]
        
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._features = self._features[perm]
          #self._features = self._features + eps
          self._labels = self._labels[perm]
          #self._cost = self._cost[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]#, self._cost[start:end]




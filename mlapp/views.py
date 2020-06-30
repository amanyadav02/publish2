from django.shortcuts import render,redirect

# Create your views here.
def home1(request,aman):
    import pandas as pd
    housing=pd.read_csv('data.csv')
    from sklearn.model_selection import train_test_split
    train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
    from sklearn.model_selection import StratifiedShuffleSplit
    split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(housing,housing['CHAS']):
        strat_train_set=housing.loc[train_index]
        strat_test_set=housing.loc[test_index]
    housing=strat_train_set.copy()
    corr_matrix=housing.corr()
    corr_matrix['MEDV'].sort_values(ascending=False)
    corr_matrix=housing.corr()
    corr_matrix['MEDV'].sort_values(ascending=False)
    housing=strat_train_set.drop("MEDV",axis=1)
    housing_labels=strat_train_set["MEDV"].copy()
    median=housing["RM"].median()
    housing["RM"].fillna(median)
    from sklearn.impute import SimpleImputer
    imputer=SimpleImputer(strategy="median")
    imputer.fit(housing)
    X=imputer.transform(housing)
    housing_tr=pd.DataFrame(X,columns=housing.columns)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    my_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('std.scalar',StandardScaler()),
        ])
    housing_num_tr=my_pipeline.fit_transform(housing_tr)
    from  sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor()
    model.fit(housing_num_tr,housing_labels)
    some_data=housing.iloc[:5]
    some_labels=housing_labels.iloc[:5]
    prepared_data=my_pipeline.transform(some_data)
    model.predict(prepared_data)
    import numpy as np
    from sklearn.metrics import mean_squared_error
    housing_predictions=model.predict(housing_num_tr)
    mse=mean_squared_error(housing_labels,housing_predictions)
    rmse=np.sqrt(mse)
    from sklearn.model_selection import cross_val_score
    scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
    rmse_scores=np.sqrt(-scores)
    from joblib import dump,load
    dump(model,'Dragon.joblib')
    x_test=strat_test_set.drop("MEDV",axis=1)
    y_test=strat_test_set["MEDV"].copy()
    x_test_prepared=my_pipeline.transform(x_test)
    final_prediction=model.predict(x_test_prepared)
    final_mse=mean_squared_error(y_test,final_prediction)
    final_rmse=np.sqrt(final_mse)
    input=np.array([aman])
    ans=model.predict(input)
    return(ans)
finalans=0
def services(request):
    if(request.method=="POST"):
        crim=request.POST.get('crim')
        zn=request.POST.get('zn')
        indus=request.POST.get('indus')
        chas=request.POST.get('chas')
        nox=request.POST.get('nox')
        rm=request.POST.get('rm')
        age=request.POST.get('age')
        dis=request.POST.get('dis')
        rad=request.POST.get('rad')
        tax=request.POST.get('tax')
        ptratio=request.POST.get('ptratio')
        lstat=request.POST.get('lstat')
        medv=request.POST.get('medv')
        aman=[int(crim),int(zn),int(indus),int(chas),int(nox),int(rm),int(age),int(dis),int(rad),int(tax),int(ptratio),int(lstat),int(medv)]
        global finalans
        finalans = home1(request,aman)
        return redirect('/home')
    else:    
        return render(request,'mlapp/services.html')
def home(request):
    context={
        'variable':finalans[-1],
            }
    return render(request,'mlapp/home.html',context)
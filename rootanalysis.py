def rootanalysis(X_data):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_data)

    apply_PCA.pca-PCA()
    apply_PCA.pca.fit(X_data)
    Xp_data=apply_PCA.pca.transform(X_data)

    components=apply_PCA.pca.components_
    explained_variance_ratio=apply_PCA.pca.explained_variance_ratio_
    explained_variance=apply_PCA.pca.explained_variance_[0.5]
    n_components=apply_PCA.pca.n_components_

    return Xp_data,components,explained_variance_ratio,explained_variance,n_components


def plot_PC_variance(X_data, explained_variance_ratio):
    Np_comp=(1,len(X_data.columns),1)
    fig,ax=plt.subplots(1,1,figsize-(10,10))
    plt.figure(4)
    ax.set_xlabel('Prinicipal Components')
    ax.set_ylabel('Explained Variance')

    ax.plot(np.arange(Np),explained_variance_ratio[0:Np],label='variance by each component')
    ax.bar(np.arange(Np),explained_variance_ratio[0:Np]='Bar plot for each component')
    ax.plot(np.arange(Np),np.cumsum(explained_variance_ratio[0:Np]),label='Cumulative variance')
    ax.legend(loc='Upper left')


def plot_pc(X_data,components):
    fig, ax=plt.subplots(1,1,figsize=(10,10))
    plt.figure(5)

    ax.set_xlabel("Principal component 1",fontsize=14)
    ax.set_ylabel('Principal component 2',fontsize=14)
    ax.set_title('Principal components 1;2',fontsize=16)

    ax.scatter(components[:,1],components[:,2])

    

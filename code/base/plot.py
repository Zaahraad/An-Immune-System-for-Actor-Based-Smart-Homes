import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns


class Plots:
    def __init__(self):
        print("plotting ...")
    
    def dimension_reduction_plot(self, df, wordEmbedding_alg, dimesionality_reduction_algorithm):
        fig = px.scatter(df, x=df.iloc[:, 0], y=df.iloc[:, 1], text="Words")
        fig.update_layout(
            width=700,   # width in pixels
            height=500,
            title=f"The result of {wordEmbedding_alg} based on {dimesionality_reduction_algorithm}",
            xaxis_title="First dimension",
            yaxis_title="Second dimension"
        )
        fig.show()
    
    def pca_Variance_Explained_plot(self, pca):
        # eigenvalues
        variance = pca.explained_variance_

        n = np.arange(1,len(variance)+1)
        plt.plot(n,variance)

        plt.ylabel('Variance Explained per PC Components (eigenvalue)')
        plt.xlabel('Num of Features(PC Components)')
        plt.title('PCA Analysis')
        plt.style.context('seaborn-whitegrid')
        plt.grid()
        plt.show()  
        
    
    def pca_spectrum_plot(self, pca):
        variance = pca.explained_variance_
        percentage_var_explained = pca.explained_variance_ratio_
        cum_var_explained = np.cumsum(percentage_var_explained)
        
        n = np.arange(1,len(variance)+1)
        
        plt.figure(figsize=(6, 4))
        plt.plot(n,cum_var_explained, linewidth=2)
        plt.axis('tight')
        plt.grid()
        plt.xlabel('n components')
        plt.ylabel('Cumulative explained variance ratio')
        plt.show()
    
    
    def clustering_plot(self, data, labels, clustering_alg):
        fig = px.scatter(data, x=data.iloc[:, 0], y=data.iloc[:, 1], color=labels , color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(
            title=f"The result of {clustering_alg} Clustrering algorithm",
            xaxis_title="First dimension",
            yaxis_title="Second dimension",
            width=700,   # width in pixels
            height=500
        )
        # for label in labels.unique():
        #     fig.update_traces(name=label, selector=dict(name=label))
        fig.update_traces(marker=dict(size=10)) 
        fig.update_traces(showlegend=True)
        fig.show()
    
    def clustering_3d_plot(self, data, clustering_alg):
        fig = px.scatter_3d(data, x=data.iloc[:, 0], y=data.iloc[:, 1], z=data.iloc[:, 2],
                    color=data.label, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(
            title=f"The result of {clustering_alg} Clustrering algorithm",
            xaxis_title="First dimension",
            yaxis_title="Second dimension",
            width=700,   # width in pixels
            height=500
        )
        
        fig.show()
    
    def values_column_plot(self, data, time, value, title=None):
        plt.figure(figsize=(20, 8))
        fig = px.line(data, x=time, y=value)
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            width=700,   # width in pixels
            height=500,
        )
        fig.show()
    
    def MSE_LSTM_plot(self, data, title):
        plt.figure(figsize=(8,6))
        plt.plot(data, label='Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Square Error')
        plt.title('Mean Square Error in Train data')
    
    def baseline_predictions_plot(self, time, original, predict_train, predict_test, yaxis_title, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, 
                                y=original,
                                name = 'orginal'))


        fig.add_trace(go.Scatter(x=time, 
                                y=predict_train,
                                name = 'train'))


        fig.add_trace(go.Scatter(x=time, 
                                y=predict_test,
                                name = 'test'))

        fig.update_layout(
            title_text=title,
            xaxis_title="Time",
            yaxis_title=yaxis_title
        )
        fig.show() 
    
    def plot_reconstruction_error(self, reconstruction_errors, type):
        # Plot the distribution of reconstruction errors
        plt.figure(figsize=(8, 6))
        plt.hist(reconstruction_errors, bins=30, density=True, alpha=0.7, color='blue', label=f'Reconstruction Errors {type}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title(f'Distribution of Reconstruction Errors ({type} Data)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def loss_plot(self, history, column, ylable, title):
        plt.figure(figsize=(8,6))
        plt.plot(history.history[column], label='Training loss')
        plt.xlabel('Epochs')
        plt.ylabel(ylable)
        plt.title(title)
        # plt.savefig('plot.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()


    def valid_train_loss_plot(self, history, title):
        # Access training history
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot loss
        plt.figure(figsize=(8, 6))
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title(title)
        # plt.savefig('MSE_LSTM_electricity_lights.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.legend()
        plt.show()
        
    def valid_train_accuracy_plot(self, history, title):
        # Access training history
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Plot accuracy (if applicable)
        plt.figure(figsize=(8, 6))
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(title)
        # plt.savefig('Accuracy.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()
    
    
    
    def plot_original_prediction(self, df, title_text, yaxis_title, name, predict_column):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.time, 
                                y=df.orginal_data,
                                name = 'orginal'))


        fig.add_trace(go.Scatter(x=df.time, 
                                y=df[predict_column],
                                name = name))


        fig.update_layout(
            title_text=title_text,
            xaxis_title="Time",
            yaxis_title=yaxis_title
        )
        # fig.write_image('LSTM_electricity_lights.png')
        fig.show()
    

    def plot_describe(self, df, save_path):
        describe_num_df = df.describe(include=['int64','float64'])
        describe_num_df.reset_index(inplace=True)
        #To remove any variable from plot
        describe_num_df = describe_num_df[describe_num_df['index'] != 'count']
        plt.figure(figsize=(8,6))
        sns.catplot(x='index', y='Value', data=describe_num_df)
        plt.title("Statistical information about Value column")
        plt.savefig(save_path, format='eps', dpi=300, bbox_inches='tight')
        plt.show() 
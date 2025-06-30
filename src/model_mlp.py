

def create_MLP(model_name, num_features, num_classes):
    '''creates MLP model'''

    model = LitGNN(model_name, model=torch_geometric.nn.models.MLP,
                    channel_list=[num_features, 128, 128, num_classes],
                    model_type='baseline')

    return model
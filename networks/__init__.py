import importlib

def find_model_using_name(model_name):
    model_filename = "networks." + model_name
    modellib = importlib.import_module(model_filename+'_net')
    create_fnc = None
    target_model_name = 'create_' + model_name + '_net'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            create_fnc = cls

    if create_fnc is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return create_fnc

def create_model(args, device):
    create_fnc = find_model_using_name(args.model_name)
    instance = create_fnc(args, args.lr, isTrain=True, device=device)
    print("model [{}] was created (rand{})".format(args.model_name, args.rank))
    return instance

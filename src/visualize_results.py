import json
import matplotlib.pyplot as plt

def visualize_causal_models():
    n_layers = 12
    
    for token in [0,1,2,3,4,5]:

        for id in [1,2,3]:

            if id == 1:
                label = '(X+Y)+Z'
            elif id == 2:
                label = '(X+Z)+Y'
            else:
                label = 'x+(Y+Z)'

            cm = []
            report_dicts = []

            for layer in range(n_layers):
                file_path = f'/home/mpislar/align-transformers/my_experiments/results_{id}/report_layer_{layer}_tkn_{token}.json'
                with open(file_path, 'r') as json_file:
                    report_dict = json.load(json_file)
                    report_dicts.append(report_dict)

            for layer, report_dict in enumerate(report_dicts, start=1):
                cm.append(report_dict['accuracy'])
            
        
            plt.scatter(range(n_layers), cm)
            plt.plot(range(n_layers), cm, label=label)
            plt.xticks(range(int(min(plt.xticks()[0])), int(max(plt.xticks()[0])) + 1))
            plt.xlabel('layer')
            plt.ylabel('IIA')

        plt.title(f'IIA when targeting token {token}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'IIA_per_layer_tkn_{token}.png')
        plt.close()

def main():
    visualize_causal_models()

if __name__ =="__main__":
    main()
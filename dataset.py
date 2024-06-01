import json
import random
from tqdm import tqdm
import copy
from datasets import Dataset, DatasetDict, load_from_disk

class ICLDataLoader:
    def __init__(self, file_path, context_size=5, dataset_size=250):
        """
        Initializes the data loader.
        
        Args:
        file_path (str): Path to the JSON file/huggingface dataset directory containing the dataset.
        context_size (int): Number of examples to include in the context prompt.
        """
        self.file_path = file_path
        self.context_size = context_size
        if 'json' in self.file_path:
            self.data = self.load_data_from_json()
        else:
            self.data = self.load_data_from_hf()
        #print(self.data)
        self.ids = list(range(len(self.data)))  # Create a list of indexes to track unused items
        self.ids_queue=copy.copy(self.ids)
        self.dataset_size = dataset_size
        random.shuffle(self.ids)  # Shuffle to ensure randomness

    def load_data_from_json(self):
        """
        Load data from a JSON file.
        
        Returns:
        List[dict]: List of dictionaries containing the antonym pairs.
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def load_data_from_hf(self):
        """
        Load data from a hugging face dataset directory. The dataset directory should be organized as follows:
        sst2
        --train
        ----data.arrow
        ----dataset_info.json
        ----state.json
        --validation
        ----...
        --dataset_dict.json

        returns:DatasetDict({
            train: Dataset({
                features: ['sentence', 'label', 'idx'],
                num_rows: 67349
            })
            validation: Dataset({
                features: ['sentence', 'label', 'idx'],
                num_rows: 872
            })
        })
        """
        data = load_from_disk(self.file_path)
        return data['train']

    def __iter__(self):
        self.current_iteration = 0
        self.tqdm_bar = tqdm(total=self.dataset_size, desc="Evalutating")
        return self

    def __next__(self):
        """
        Generates the next in-context prompt, cycling through the data list, until every word has been used as the test word, then repeat
        
        Returns:
        str: The generated in-context prompt.
        """   
        # Pick the next item for prediction and remove it from the list
        if self.current_iteration >= self.dataset_size:
            self.tqdm_bar.close()  # Close the tqdm bar once done
            raise StopIteration

        if len(self.ids_queue)==0:
            random.shuffle(self.ids)  # Shuffle to ensure randomness
            self.ids_queue=copy.copy(self.ids)
            
        next_index = self.ids_queue.pop(0)
        prediction_item = self.data[next_index]
        context_candidates = [item for item in self.data if item != prediction_item]
        # Select context examples
        context_examples = random.sample(context_candidates, self.context_size)
        
        # Build the prompt
        if 'json' in self.file_path:
            prompt = "\n".join([f"{item['input']}: {item['output']}" for item in context_examples])
            prompt += f"\n{prediction_item['input']}: "
        else:
            prompt = "\n".join([f"Sentence: {item['sentence']} Answer: {item['label']}" for item in context_examples])
            prompt += f"\nSentence: {prediction_item['sentence']} Answer: "
        self.tqdm_bar.update(1)
        self.current_iteration += 1
        return prompt

# Example usage:
# Assuming you have 'Final_Antonym_Dataset.json' in the correct directory.
#data_loader = ICLDataLoader('/home/pc/data/qixun/datasets/NLP_datasets/sentiment100.json', context_size=5, dataset_size=250)
#for i, p in enumerate(data_loader):
    #print(p)
    #print('\n')
#    pass


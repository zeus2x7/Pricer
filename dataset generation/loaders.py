from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures  import ProcessPoolExecutor, ThreadPoolExecutor
from items import Item 
from huggingface_hub import login 

CHUNK_SIZE = 1000
MIN_PRICE =0.5
MAX_PRICE = 999.49
hf_token = ""
login(hf_token, add_to_git_credential=True)
class ItemLoader:
    
    def __init__(self, name):
        self.name = name 
        self.dataset = None 

    def from_datapoint(self, datapoint):

        try:
            price_Str = datapoint['price']
            if price_Str:
                price = float(price_Str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return  item if item.include else None 
        except ValueError:
            return None
    
    def from_chunk(self, chunk):

        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result: 
                batch.append(result)
        return batch 
    
    def chunk_generator(self):

        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i,(min(i + CHUNK_SIZE, size))))
    
    def load_in_parallel(self, workers):

        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) +1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total = chunk_count):
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results 
    
    def load(self, workers = 8):

        start = datetime.now()
        print(f"Loading dataset {self.name}", flush = True)
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.name}", split ="full", trust_remote_code=True)
        #self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split = "full", trust_remote_code= True)
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"Completed {self.name} with {len(results):} datapoints in {(finish-start).total_seconds()/60:} mins", flush=True)
        return results



        
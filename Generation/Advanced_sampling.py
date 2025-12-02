import torch
import torch.nn.functional as F

class Sampler:    
    @staticmethod
    def apply_temperature(logits, temperature):
        return logits / temperature

    @staticmethod
    def apply_top_k(logits, k):
        if k is None or k == 0:
            return logits
            
        top_val, _ = torch.topk(logits, k)
        min_top_val = top_val[:, -1].unsqueeze(-1)
        
        logits[logits < min_top_val] = float('-inf')
        return logits

    @staticmethod
    def apply_top_p(logits, p):
        if p is None or p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits

    @staticmethod
    def get_next_token(logits, temperature=1.0, top_k=None, top_p=None):
        logits = logits[:, -1, :] 
        logits = Sampler.apply_temperature(logits, temperature)
        
        if top_k is not None:
            logits = Sampler.apply_top_k(logits, top_k)
            
        if top_p is not None:
            logits = Sampler.apply_top_p(logits, top_p)
            
        probs = F.softmax(logits, dim=-1)
        
        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
            
        return next_token

    def beam_search_decoder(logits_function, start_token, beam_width=3, max_steps=5):
        sequences = [[([start_token], 0.0)]]
        
        for _ in range(max_steps):
            all_candidates = []
            
            for seq, score in sequences:
                next_probs = torch.softmax(torch.randn(10), dim=-1)
                
                for i in range(len(next_probs)):
                    candidate = (seq + [i], score - torch.log(next_probs[i]).item())
                    all_candidates.append(candidate)
            
            ordered = sorted(all_candidates, key=lambda tup: tup[1])
            sequences = ordered[:beam_width]
            
        return sequences

if __name__ == "__main__":
    torch.manual_seed(42)
    
    dummy_logits = torch.tensor([[1.0, 2.0, 5.0, 0.5, 0.1, 0.1, 0.1, 4.0, 0.2, 0.3]])
    
    print("Original Logits:", dummy_logits)
    
    print("\nHigh Temp (2.0):", Sampler.apply_temperature(dummy_logits.clone(), 2.0))
    
    print("\nTop-k (k=3):", Sampler.apply_top_k(dummy_logits.clone(), k=3))
    
    print("\nTop-p (p=0.8):", Sampler.apply_top_p(dummy_logits.clone(), p=0.8))
"""
src/model_a_neural.py

As an alternative to OHE/TF-IDF feature engineering, this script shows 
how you could use a Pre-trained Neural Language Model (Sentence Transformers)
to compute embeddings and verify RACE reading comprehension answers.
NOTE: This takes much longer and requires GPU for good performance.
Hence it is entirely optional.
"""

def get_neural_verification_score(article: str, question: str, option: str) -> float:
    # We dynamically import so it doesn't fail if the user only has sklearn installed
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
    except ImportError:
        print("Please run `pip install sentence-transformers torch` to use the neural baseline.")
        return 0.0

    print("Loading lightweight neural baseline model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # we encode the article as context, and Q+A as the hypothesis
    context = article
    hypothesis = f"Question: {question} Answer: {option}"

    emb_ctx = model.encode(context, convert_to_tensor=True)
    emb_hyp = model.encode(hypothesis, convert_to_tensor=True)

    # cosine sim between the whole text and the qa pair
    sim = util.pytorch_cos_sim(emb_ctx, emb_hyp).item()
    return float(sim)

if __name__ == "__main__":
    print("Neural Baseline demo started.")
    demo_art = "The Eiffel Tower is located in Paris, France. It is a famous landmark."
    demo_q = "Where is the Eiffel Tower?"
    
    print(f"Article: {demo_art}")
    print(f"Question: {demo_q}")
    print("Option A: In London")
    print(f"Score A: {get_neural_verification_score(demo_art, demo_q, 'In London'):.4f}")
    print("Option B: In Paris")
    print(f"Score B: {get_neural_verification_score(demo_art, demo_q, 'In Paris'):.4f}")

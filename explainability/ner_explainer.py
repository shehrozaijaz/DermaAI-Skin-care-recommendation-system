import torch
import numpy as np

class NERExplainer:
    """Explains NER predictions by computing token-level importance scores."""
    
    def __init__(self, model, word2idx, idx2tag):
        self.model = model
        self.word2idx = word2idx
        self.idx2tag = idx2tag
        self.model.eval()
    
    def explain_prediction(self, sentence_tensor, words):
        """
        Generate explanation for a prediction.
        
        Args:
            sentence_tensor: Input tensor [1, seq_len]
            words: List of words in the sentence
            
        Returns:
            Dictionary with token importances and predictions
        """
        # Get base prediction
        with torch.no_grad():
            predictions = self.model(sentence_tensor)[0]
        
        # Compute gradient-based importance
        importance_scores = self._compute_gradient_importance(sentence_tensor)
        
        # Get LSTM hidden states for attention visualization
        attention_weights = self._extract_attention_weights(sentence_tensor)
        
        # Build explanation
        explanations = []
        for i, (word, pred_tag_idx) in enumerate(zip(words, predictions)):
            if i >= len(importance_scores):
                break
                
            explanations.append({
                'word': word,
                'predicted_tag': self.idx2tag.get(pred_tag_idx, 'O'),
                'importance': float(importance_scores[i]),
                'attention': float(attention_weights[i]) if i < len(attention_weights) else 0.0
            })
        
        return {
            'tokens': explanations,
            'overall_confidence': float(np.mean([e['importance'] for e in explanations]))
        }
    
    def _compute_gradient_importance(self, sentence_tensor):
        """Compute importance scores using gradient-based attribution (Saliency)."""
        # 1. Get embeddings and detach to treat as leaf node for gradients
        embeds = self.model.embedding(sentence_tensor).detach()
        embeds.requires_grad = True
        
        # 2. Run forward pass from embeddings
        lstm_out, _ = self.model.lstm(embeds)
        emissions = self.model.hidden2tag(lstm_out)
        
        # 3. Compute score (sum of max emissions for simplicity)
        # We want to know which tokens contributed most to the high scores
        max_emissions, _ = emissions.max(dim=-1)
        score = max_emissions.sum()
        
        # 4. Backpropagate
        self.model.zero_grad()
        score.backward()
        
        # 5. Compute importance (L2 norm of gradients)
        # embeds.grad shape: [1, seq_len, embed_dim]
        gradients = embeds.grad
        importance = gradients.norm(dim=-1).squeeze(0).numpy()
        
        # Normalize to 0-1
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        elif importance.max() > 0:
            importance = importance / importance.max()
            
        return importance
    
    def _extract_attention_weights(self, sentence_tensor):
        """Extract attention-like weights from LSTM hidden states."""
        with torch.no_grad():
            embeds = self.model.embedding(sentence_tensor)
            lstm_out, _ = self.model.lstm(embeds)
            
            # Use L2 norm of hidden states as attention proxy
            attention = torch.norm(lstm_out, dim=-1).squeeze(0).numpy()
            
            # Normalize
            if attention.max() > 0:
                attention = attention / attention.max()
            
            return attention
    
    def get_entity_explanation(self, words, entities, predictions):
        """
        Explain why specific entities were detected.
        
        Args:
            words: List of words
            entities: List of detected entities
            predictions: List of predicted tag indices
            
        Returns:
            Dictionary mapping entities to their contributing tokens
        """
        entity_explanations = {}
        
        current_entity = []
        current_entity_words = []
        
        for word, tag_idx in zip(words, predictions):
            tag = self.idx2tag.get(tag_idx, 'O')
            
            if tag.startswith('B-'):
                if current_entity:
                    entity_text = ' '.join(current_entity_words)
                    if entity_text in entities:
                        entity_explanations[entity_text] = {
                            'tokens': current_entity,
                            'tag_type': current_entity[0]['tag'].split('-')[1] if '-' in current_entity[0]['tag'] else 'UNKNOWN'
                        }
                current_entity = [{'word': word, 'tag': tag}]
                current_entity_words = [word]
            elif tag.startswith('I-') and current_entity:
                current_entity.append({'word': word, 'tag': tag})
                current_entity_words.append(word)
            else:
                if current_entity:
                    entity_text = ' '.join(current_entity_words)
                    if entity_text in entities:
                        entity_explanations[entity_text] = {
                            'tokens': current_entity,
                            'tag_type': current_entity[0]['tag'].split('-')[1] if '-' in current_entity[0]['tag'] else 'UNKNOWN'
                        }
                current_entity = []
                current_entity_words = []
        
        # Handle last entity
        if current_entity:
            entity_text = ' '.join(current_entity_words)
            if entity_text in entities:
                entity_explanations[entity_text] = {
                    'tokens': current_entity,
                    'tag_type': current_entity[0]['tag'].split('-')[1] if '-' in current_entity[0]['tag'] else 'UNKNOWN'
                }
        
        return entity_explanations

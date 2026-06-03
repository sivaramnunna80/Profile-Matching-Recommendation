import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class MatchingEngine:
    def __init__(self,datapath):
        self.datapath=datapath
        self.df=pd.read_csv(self.datapath)
        self.w1=0.5 #For Text
        self.w2=0.3 #For MBTI
        self.w3=0.2 #For Location
    
    def text_similiarity(self):
        tfidf=TfidfVectorizer()
        vectors=tfidf.fit_transform(self.df['cleaned_text'])
        return vectors
    
    def similarity_text_score(self,idx1,idx2):
        vectors=self.text_similiarity()
        user1=vectors[idx1]
        user2=vectors[idx2]
        score=cosine_similarity(user1,user2)[0][0]
        return score
    
    def location_score(self,loc1,loc2):
        score=0
        if loc1==loc2:
            score=1
        return score
    
    def mbti_score(self,mbti1,mbti2):
        score=0
        if mbti1==mbti2:
            score=1
        else:
            if mbti1[0]==mbti2[0]:
                score=score+0.15
            if mbti1[1]==mbti2[1]:
                score=score+0.15
            if mbti1[2]==mbti2[2]:
                score=score+0.15
            if mbti1[3]==mbti2[3]:
                score=score+0.15
        return score
    
    def update_weights_adaptive(self, features, action_target, learning_rate=0.05):
        """
        Gradient Descent Adaptive Feedback Loop Update.
        
        Parameters:
        -----------
        features : np.array([text_sim, mbti_sim, loc_sim])
            Calculated matrix component overlap similarities (each between 0 and 1)
        action_target : float
            1.0 for profile 'Accept', 0.0 for profile 'Reject'
        """
        import numpy as np
        
        # Load active vector weights
        w = np.array([self.w1, self.w2, self.w3])
        
        # Calculate localized system predictions 
        prediction = np.dot(w, features) / np.sum(w)
        error = prediction - float(action_target)
        
        # Update via localized gradient vectors step
        gradient = error * features
        updated_weights = w - (learning_rate * gradient)
        
        # Clip values to guarantee weights do not drop below zero or turn entirely negative
        updated_weights = np.clip(updated_weights, 0.05, 1.0)
        
        # Re-assign state updates back to core properties
        self.w1 = float(updated_weights[0])
        self.w2 = float(updated_weights[1])
        self.w3 = float(updated_weights[2])
        
        return updated_weights
    
    def total_score(self,user1,user2):
        row1=self.df[self.df["user_id"]==user1]
        row2=self.df[self.df["user_id"]==user2]
        
        if row1.empty or row2.empty:
            return None
        
        user1_idx=row1["user_id"].index[0]
        user2_idx=row2["user_id"].index[0]
        
        user1_mbti=row1['mbti_type'].iloc[0]
        user2_mbti=row2['mbti_type'].iloc[0]
        
        user1_loc=row1['location'].iloc[0]
        user2_loc=row2['location'].iloc[0]
        
        txt_score=self.similarity_text_score(user1_idx,user2_idx)
        mbti_type_score=self.mbti_score(user1_mbti,user2_mbti)
        loc_score=self.location_score(user1_loc,user2_loc)
        
        final_score=(self.w1*txt_score + self.w2*mbti_type_score + self.w3*loc_score)*100
        final_score=float(round(final_score,2))
        return final_score
    
    def get_top_matches(self,user_id,top_n=5):
        matches=[]
        
        for other_users in self.df['user_id']:
            
            if user_id==other_users:
                continue
            
            score=self.total_score(user_id,other_users)
            if score is not None:
                matches.append((other_users,score))
        matches = sorted(matches,key=lambda x: x[1],reverse=True)
        return matches[:top_n]
        
if __name__=="__main__":
    datapath="cleaned_dataset.csv"
    e=MatchingEngine(datapath)
    
    
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

# # ----------------------------------------------------
# # Static MBTI Pair Matching Dictionary (Sample Rule Matrix)
# # ----------------------------------------------------
# MBTI_MATRIX = {
#     ('INTJ', 'ENFP'): 1.00, ('INTJ', 'ENTJ'): 0.90, ('INTJ', 'INTP'): 0.85,
#     ('ENFP', 'INFJ'): 1.00, ('ENFP', 'INTJ'): 1.00, ('ENFP', 'ENFP'): 0.70,
#     ('INFJ', 'ENFJ'): 0.95, ('INFP', 'ENFJ'): 0.95, ('ISTJ', 'ESFJ'): 0.90,
# }

# def get_mbti_score(mbti_a, mbti_b):
#     """Returns a normalized score (0 to 1) for an MBTI pairing."""
#     pair = (mbti_a, mbti_b)
#     inv_pair = (mbti_b, mbti_a)
#     if pair in MBTI_MATRIX:
#         return MBTI_MATRIX[pair]
#     elif inv_pair in MBTI_MATRIX:
#         return MBTI_MATRIX[inv_pair]
#     return 0.50 # Default baseline compatibility score

# class AdaptiveMatcher:
#     def __init__(self, users_df, learning_rate=0.05):
#         self.users_df = users_df.copy()
#         self.lr = learning_rate
        
#         # User customized weights initialized uniformly
#         # w1 = Text Similarity, w2 = MBTI Match, w3 = Location Match
#         self.user_weights = {
#             row['user_id']: np.array([0.4, 0.4, 0.2]) for _, row in self.users_df.iterrows()
#         }
        
#         # Precompute TF-IDF vector embeddings for bios/summaries
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         combined_text = self.users_df['professional_summary'].fillna('') + " " + self.users_df['about_me'].fillna('')
#         self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)

#     def calculate_compatibility(self, user_a_id, user_b_id):
#         """Calculates a dynamic weighted compatibility score from 0 to 100%."""
#         idx_a = self.users_df[self.users_df['user_id'] == user_a_id].index[0]
#         idx_b = self.users_df[self.users_df['user_id'] == user_b_id].index[0]
        
#         user_a = self.users_df.iloc[idx_a]
#         user_b = self.users_df.iloc[idx_b]

#         # 1. Text Similarity Score (NLP Module)
#         vec_a = self.tfidf_matrix[idx_a]
#         vec_b = self.tfidf_matrix[idx_b]
#         text_sim = cosine_similarity(vec_a, vec_b)[0][0]

#         # 2. MBTI Similarity Score (Logic Layer)
#         mbti_sim = get_mbti_score(user_a['mbti'], user_b['mbti'])

#         # 3. Location Component (Binary match weight)
#         loc_sim = 1.0 if str(user_a['location']).strip().lower() == str(user_b['location']).strip().lower() else 0.0

#         # Retrieve specific tracking weights for user_a
#         weights = self.user_weights[user_a_id]
        
#         # Compute base score components matrix
#         features = np.array([text_sim, mbti_sim, loc_sim])
        
#         # Calculated via Formula: TotalScore = (w1 * TextSim) + (w2 * MBTIMatch) + (w3 * Location)
#         raw_score = np.dot(weights, features)
        
#         # Normalize sum of weights gracefully to display an exact 0-100 percentage range
#         normalized_score = (raw_score / np.sum(weights)) * 100
#         return round(normalized_score, 2), features

#     def update_weights(self, user_id, matched_user_id, action):
#         """
#         Adaptive Online Feedback Optimization via Gradient Descent Logic.
#         Updates user preference weights dynamically matching user intentions.
#         """
#         # Calculate current features
#         _, features = self.calculate_compatibility(user_id, matched_user_id)
        
#         current_weights = self.user_weights[user_id]
#         prediction = np.dot(current_weights, features) / np.sum(current_weights)
        
#         # Target action map: Accept = 1.0, Reject = 0.0
#         target = float(action)
#         error = prediction - target
        
#         # Update components through gradient tracking step
#         gradient = error * features
#         new_weights = current_weights - (self.lr * gradient)
        
#         # Prevent completely negative or zero feature degradation thresholds
#         self.user_weights[user_id] = np.clip(new_weights, 0.05, 1.0)
#         return self.user_weights[user_id]
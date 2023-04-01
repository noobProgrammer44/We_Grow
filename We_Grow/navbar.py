import streamlit as st
import numpy as np
from googleapiclient.discovery import build
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from PIL import Image

st.set_page_config(layout="wide")

height_x = 1000
width_x= 990
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

catg = {
    1 : 'Film & Animation' ,
    2 : 'Autos & Vehicles',
    10 : 'Music',
    15 : 'Pets & Animals',
    17 : 'Sports',
    18 : 'Short Movies',
    19 : 'Travel & Events',
    20 : 'Gaming',
    21 : 'Videoblogging',
    22 : 'People & Blogs',
    23 : 'Comedy',
    24 : 'Entertainment',
    25 : 'News & Politics',
    26 : 'Howto & Style',
    27 : 'Education',
    28 : 'Science & Technology',
    29 : 'Nonprofits & Activism',
    30 : 'Movies',
    31 : 'Anime/Animation',
    32 : 'Action/Adventure',
    33 : 'Classics',
    34 : 'Comedy',
    35 : 'Documentary',
    36 : 'Drama',
    37 : 'Family',
    38 : 'Foreign',
    39 : 'Horror',
    40 : 'Sci-Fi/Fantasy',
    41 : 'Thriller',
    42 : 'Shorts',
    43 : 'Shows',
    44 : 'Trailers'
}
global_concat=0

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1
def get_video_details(youtube, video_ids):
    all_video_stats = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=','.join(video_ids[i:i+50]))
        response = request.execute()
        
        for video in response['items']:
            video_stats = dict(
                            #    Duration = video['contentDetails']['duration'],
                               Title = video['snippet']['title'],
                               Description = video['snippet']['description'],
                               Thumbnail_url = video['snippet']['thumbnails']['default']['url'],
                               Published_date = video['snippet']['publishedAt'],
                               Category_id = video['snippet']['categoryId'],
                               Views = video['statistics']['viewCount'],
                               Likes = video['statistics']['likeCount'],
                               Comments = video['statistics']['commentCount']
                              
                               )
            all_video_stats.append(video_stats)
    
    return all_video_stats

def streamlit_menu(example):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Comparitive Study", "Viewer Behaviour Analysis","Recommendation Page","Dashboard Analytics"],  # required*
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":

   HtmlFile = open("index.html", 'r', encoding='utf-8')
   source_code = HtmlFile.read() 
   components.html(source_code,height=height_x,width=width_x)

   
if selected == "Comparitive Study":
    txt_input = st.text_input('Enter Your Video Link', '')
    if(txt_input):
        st.write('Now Select your Category for ',txt_input)
        text = txt_input
        uri = text.split('=')
        url=uri[1]
        api_key = 'AIzaSyBSWarpbANFLwHqmh2Q3yfz-UW7EMhJMYg'
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_ids = [url]
        video_details = get_video_details(youtube, video_ids)
        video_data = pd.DataFrame(video_details)
        x=int(video_data['Category_id'][0])
        video_data['Category_id']=catg[x]
        st.write(video_data)
        global_concat=video_data








    category=['People & Blogs','Science & Technology', 'Nonprofits & Activism','Entertainment','Education','News & Politics','Music','Travel & Events','Film & Animation','Sports','Gaming','Comedy','Howto & Style']
    st.title("Comparitive Study")
    categorys = st.selectbox('Select your Category',options=category)
    import pandas as pd 
    import numpy as np 
    import seaborn as sns 
    import matplotlib.pyplot as plt
    df_geeks = pd.read_csv("Trending videos on youtube dataset.csv")
    df_geeks.drop(['channelId','videoId','videoCategoryId','duration','caption'],axis=1,inplace=True)
    df_geeks['publishedAt']=df_geeks['publishedAt'].apply(pd.to_datetime)
    df_geeks["month"]=df_geeks['publishedAt'].dt.month
    df_geeks["year"]=df_geeks['publishedAt'].dt.year
    df_geeks['date']=df_geeks['publishedAt'].dt.day
    df_geeks.drop(['publishedAt'],axis=1,inplace=True)
    values=['viewCount','likeCount','commentCount']
    value_test=["Views","Likes","Comments"]
    def product_category(cat,field):
        fig,ax=plt.subplots(figsize=(20,10))
        ax.tick_params(axis='x', rotation=90)
        ax = sns.barplot(x='channelTitle', y=field,data=df_geeks[df_geeks['videoCategoryLabel'] == cat])
        st.pyplot(fig)
    categories=['People & Blogs','Science & Technology', 'Nonprofits & Activism','Entertainment','Education','News & Politics','Music','Travel & Events','Film & Animation','Sports','Gaming','Comedy','Howto & Style']
    index=categories.index(categorys)
    for value in values:
        product_category(categories[index],value)
    for value in value_test:
        fig,ax=plt.subplots(figsize=(20,10))
        ax.tick_params(axis='x', rotation=90)
        ax = sns.countplot(x=value,data=global_concat)
        st.pyplot(fig)
        
        # fig,ax=plt.subplots(figsize=(20,10))
        # ax.tick_params(axis='x', rotation=90)
        # ax = sns.barplot(x='Title', y=value,data=global_concat)
        # st.pyplot(fig)

if selected == "Content Analysis":
    my_input = st.text_input('Enter Your Video Link', '')
    my_url = my_input.split('=')
    my_url=my_url[1]
    #if(txt_input):
    comp_input = st.text_input('Enter Your Competetitor Video Link', '')
    comp_url = comp_input.split('=')
    comp_url=comp_url[1]







# for value in values:
#   product_category(value)



    
if selected == "Recommendation Page":
    
    st.title(f"You have selected {selected}")
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    movie_names = movies_df.set_index('movieId')['title'].to_dict()
    n_users = len(ratings_df.userId.unique())
    n_items = len(ratings_df.movieId.unique())
    
    class MatrixFactorization(torch.nn.Module):
    
        def __init__(self, n_users, n_items, n_factors=20):
            super().__init__()
            # create user embeddings
            self.user_factors = torch.nn.Embedding(n_users, n_factors) # think of this as a lookup table for the input.
            # create item embeddings
            self.item_factors = torch.nn.Embedding(n_items, n_factors) # think of this as a lookup table for the input.
            self.user_factors.weight.data.uniform_(0, 0.05)
            self.item_factors.weight.data.uniform_(0, 0.05)
            
        def forward(self, data):
            # matrix multiplication
            users, items = data[:,0], data[:,1]
            return (self.user_factors(users)*self.item_factors(items)).sum(1)
        # def forward(self, user, item):
        # 	# matrix multiplication
        #     return (self.user_factors(user)*self.item_factors(item)).sum(1)
        
        def predict(self, user, item):
            return self.forward(user, item)

    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader # package that helps transform your data to machine learning readiness

# Note: This isn't 'good' practice, in a MLops sense but we'll roll with this since the data is already loaded in memory.
    class Loader(Dataset):
        def __init__(self):
            self.ratings = ratings_df.copy()
            
            # Extract all user IDs and movie IDs
            users = ratings_df.userId.unique()
            movies = ratings_df.movieId.unique()
            
            #--- Producing new continuous IDs for users and movies ---
            
            # Unique values : index
            self.userid2idx = {o:i for i,o in enumerate(users)}
            self.movieid2idx = {o:i for i,o in enumerate(movies)}
            
            # Obtained continuous ID for users and movies
            self.idx2userid = {i:o for o,i in self.userid2idx.items()}
            self.idx2movieid = {i:o for o,i in self.movieid2idx.items()}
            
            # return the id from the indexed values as noted in the lambda function down below.
            self.ratings.movieId = ratings_df.movieId.apply(lambda x: self.movieid2idx[x])
            self.ratings.userId = ratings_df.userId.apply(lambda x: self.userid2idx[x])
            
            
            self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
            self.y = self.ratings['rating'].values
            self.x, self.y = torch.tensor(self.x), torch.tensor(self.y) # Transforms the data to tensors (ready for torch models.)

        def __getitem__(self, index):
            return (self.x[index], self.y[index])

        def __len__(self):
            return len(self.ratings)
    num_epochs = 128
    cuda = torch.cuda.is_available()

    print("Is running on GPU:", cuda)

    model = MatrixFactorization(n_users, n_items, n_factors=8)
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # GPU enable if you have a GPU...
    if cuda:
        model = model.cuda()

    # MSE loss
    loss_fn = torch.nn.MSELoss()

    # ADAM optimizier
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train data
    train_set = Loader()
    train_loader = DataLoader(train_set, 128, shuffle=True)

    for it in tqdm(range(num_epochs)):
        losses = []
        for x, y in train_loader:
            if cuda:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
    c = 0
    uw = 0
    iw = 0 
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            if c == 0:
                uw = param.data
                c +=1
            else:
                iw = param.data
    trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()
    from sklearn.cluster import KMeans
    # Fit the clusters based on the movie weights
    kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)
    for cluster in range(10):
        print("Cluster #{}".format(cluster))
        movs = []
        for movidx in np.where(kmeans.labels_ == cluster)[0]:
            movid = train_set.idx2movieid[movidx]
            rat_count = ratings_df.loc[ratings_df['movieId']==movid].count()[0]
            movs.append((movie_names[movid], rat_count))
        for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
            print("\t", mov[0])
    clust=[]
    for cluster in range(10):
        print("Cluster #{}".format(cluster))
        movs = []
        for movidx in np.where(kmeans.labels_ == cluster)[0]:
            movid = train_set.idx2movieid[movidx]
            rat_count = ratings_df.loc[ratings_df['movieId']==movid].count()[0]
            movs.append((movie_names[movid], rat_count))
        for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
            print("\t", mov[0])
            clust.append(mov[0])
    # category=['People & Blogs','Science & Technology', 'Nonprofits & Activism','Entertainment','Education','News & Politics','Music','Travel & Events','Film & Animation','Sports','Gaming','Comedy','Howto & Style']
    st.title("Select a video from the below")
    dropbox = st.selectbox('Select your Category',options=clust)

    test_dict = {'1' : ['Star Wars: Episode IV - A New Hope (1977)','Terminator 2: Judgment Day (1991)','Shrek (2001)',
    'Men in Black (a.k.a. MIB) (1997)',
    'Memento (2000)',
    'Dark Knight, The (2008)',
    'Babe (1995)',
    'Truman Show, The (1998)',
    'Breakfast Club, The (1985)',
    'Net, The (1995)'], 
    '2' : [ 'Lord of the Rings: The Fellowship of the Ring, The (2001)',
    'Lord of the Rings: The Return of the King, The (2003)',
    'Sixth Sense, The (1999)',
    'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
    'Mask, The (1994)',
    'Beauty and the Beast (1991)',
    'Inception (2010)',
    'Good Will Hunting (1997)',
    'GoldenEye (1995)',
    'Terminator, The (1984)'], 
    '3':[ 'Toy Story (1995)',
    'American Beauty (1999)',
    'Batman (1989)',
    'Alien (1979)',
    'Groundhog Day (1993)',
    'Stargate (1994)',
    'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
    'Monsters, Inc. (2001)',
    'Reservoir Dogs (1992)',
    'E.T. the Extra-Terrestrial (1982)'],
    '4' :['Fight Club (1999)',
    'Seven (a.k.a. Se7en) (1995)',
    'Apollo 13 (1995)',
    'Fugitive, The (1993)',
    'Fargo (1996)',
    'Pirates of the Caribbean: The Curse of the Black Pearl (2003)',
    'Die Hard (1988)',
    'Mrs. Doubtfire (1993)',
    'Kill Bill: Vol. 1 (2003)',
    'American History X (1998)'],
    '5':['Forrest Gump (1994)',
    'Silence of the Lambs, The (1991)',
    'Matrix, The (1999)',
    'Star Wars: Episode VI - Return of the Jedi (1983)',
    'Mission: Impossible (1996)',
    'Titanic (1997)',
    "One Flew Over the Cuckoo's Nest (1975)",
    'Godfather: Part II, The (1974)',
    'Goodfellas (1990)',
    'Clockwork Orange, A (1971)'],
    '6':['Pulp Fiction (1994)',
    'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
    'Aladdin (1992)',
    'True Lies (1994)',
    'Lion King, The (1994)',
    'Star Wars: Episode I - The Phantom Menace (1999)',
    'Batman Forever (1995)',
    'Monty Python and the Holy Grail (1975)',
    'Twister (1996)',
    "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)"],
    '7':['Jurassic Park (1993)',
    'Braveheart (1995)',
    "Schindler's List (1993)",
    'Usual Suspects, The (1995)',
    'Back to the Future (1985)',
    'Speed (1994)',
    'Gladiator (2000)',
    'Dances with Wolves (1990)',
    'Princess Bride, The (1987)',
    'Finding Nemo (2003)'],
    '8':['Independence Day (a.k.a. ID4) (1996)',
    'Saving Private Ryan (1998)',
    'Lord of the Rings: The Two Towers, The (2002)',
    'Pretty Woman (1990)',
    'X-Men (2000)',
    'Incredibles, The (2004)',
    'Big Lebowski, The (1998)',
    'Outbreak (1995)',
    'Office Space (1999)',
    'Star Wars: Episode II - Attack of the Clones (2002)'],
    '9' : [ 'Shawshank Redemption, The (1994)',
    'Star Wars: Episode V - The Empire Strikes Back (1980)',
    'Godfather, The (1972)',
    'Ace Ventura: Pet Detective (1994)',
    'Indiana Jones and the Last Crusade (1989)',
    'Aliens (1986)',
    'Beautiful Mind, A (2001)',
    "Ocean's Eleven (2001)",
    'Bourne Identity, The (2002)',
    'Green Mile, The (1999)']
    }
    key_list=list(test_dict.keys())
    val_list=list(test_dict.values())
    for i in range(len(val_list)):
        for x in range(len(val_list)):
            if (dropbox==val_list[i][x]):
                final_ans=val_list[i]
                break
    st.write("If you like the above video, below is the recommended list of videos you would like to watch")
    st.write(final_ans)

    




if selected == "Viewer Behaviour Analysis":
    st.title(f"You have selected {selected}")
    values = ['Sub­scribers lost', 'Sub­scribers gained', 'RPM (USD)', 'CPM (USD)',
       'Av­er­age per­cent­age viewed (%)',
       'Views', 'Watch time (hours)', 'Sub­scribers',
       'Your es­tim­ated rev­en­ue (USD)', 'Im­pres­sions',
       'Im­pres­sions click-through rate (%)']
    import pandas as pd 
    import numpy as np 
    import seaborn as sns 
    import matplotlib.pyplot as plt
    df_result = pd.read_csv("Aggregated_Metrics_By_Video.csv")
    dfresult = df_result.dropna(subset=['Video title','Video pub­lish time','CPM (USD)'])
    dfresult["Sub­scribers"] = np.where(dfresult["Sub­scribers"]<=0,0,dfresult["Sub­scribers"])
    df_res=df_result.drop(df_result.tail(191).index)
    def product_category(field):
        fig,ax=plt.subplots(figsize=(20,10))
        ax.tick_params(axis='x', rotation=90)
        ax = sns.barplot(x='Video title',y=field,data=df_res)
        st.pyplot(fig)
    for x in values:
        product_category(x)

if selected == "Youtube Content Analysis":
    st.title(f"You have selected {selected}")
    txt_input = st.text_input('Enter Your Video Link', '')
    text = txt_input
    uri = text.split('=')
    url=uri[1]
    api_key = 'AIzaSyBSWarpbANFLwHqmh2Q3yfz-UW7EMhJMYg'
    youtube = build('youtube', 'v3', developerKey=api_key)
    def get_video_details(youtube, video_ids):
        all_video_stats = []
        
        for i in range(0, len(video_ids), 50):
            request = youtube.videos().list(
                        part='snippet,statistics,contentDetails',
                        id=','.join(video_ids[i:i+50]))
            response = request.execute()
            
            for video in response['items']:
                video_stats = dict(
                                Duration = video['contentDetails']['duration'],
                                Title = video['snippet']['title'],
                                Description = video['snippet']['description'],
                                Thumbnail_url = video['snippet']['thumbnails']['default']['url'],
                                Definition = video['contentDetails']['definition'],
                                Caption = video['contentDetails']['caption'],
                                Categoryid = video['snippet']['categoryId'],
                                Tags = video['snippet']['tags'],
                                Views = video['statistics']['viewCount'],
                                Likes = video['statistics']['likeCount'],
                                Comments = video['statistics']['commentCount']
                                #  Channel_id = video['snippet']['channelId']
                                
                                )
                all_video_stats.append(video_stats)
        
        return all_video_stats
    video_ids = [url]
    video_details = get_video_details(youtube, video_ids)   
    video_data = pd.DataFrame(video_details)
    
    # video_data['Categoryid'] = pd.to_numeric(video_data['Categoryid'])
    video_data['Views'] = pd.to_numeric(video_data['Views'])
    video_data['Likes'] = pd.to_numeric(video_data['Likes'])
    # video_data['Duration'] =pd.to_numeric(video_data['Duration'])
    video_data['Comments'] = pd.to_numeric(video_data['Comments'])
    st.write(video_data)








if selected == "Dashboard Analytics":
    st.title(f"You have selected {selected}")
    image = Image.open('dash.png')
    st.image(image, caption='Analytics')


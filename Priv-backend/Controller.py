
# coding: utf-8

# In[2]:

from summa import summarizer
import torch
import torch.nn as nn
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.tokenize import sent_tokenize
import textstat
import json


# In[ ]:




# In[3]:

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# In[4]:


ag_news_label = {
1: 'First Party Collection/Use',
2: 'Third Party Sharing/Collection',
3: 'User Choice/Control',
4: 'Data Security',
5: 'International and Specific Audiences',
6: 'User Access, Edit and Deletion',
7: 'Policy Change',
8: 'Data Retention',
9: 'Do Not Track',
10: 'Other'
                 }

NER_Categories={'court' : ["court", "jurisdiction"],
                'article':[ "law", "article"],
                'legality':[ 'legal', 'illegal', 'violates', 'terms and conditions'],
                'activity':[ 'Activity', 'Activities']
               }

dict_compare_sitename_to_filename={
    "google":"Compare_google_JSON",
    "amazon":"Compare_amazon_JSON",
    "youtube":"Compare_youtube_JSON",
    "facebook":"Compare_facebook_JSON"
    
}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


# In[ ]:




# In[5]:

model = TextSentiment(203805,32,10)
model.load_state_dict(torch.load('Classifier_model_dict.pt'))
model.eval()

vocab=torch.load('model_vocab_dict.pt')



# In[ ]:




# In[ ]:




# In[6]:

unwanted_sentences=[',','.','']
unwanted_words=['</SECTION>','<SECTION>','</POLICY>','<POLICY>','</SUBTEXT>','<SUBTEXT/>','<SUBTEXT>','<SUBTITLE>','</SUBTITLE>']



# In[7]:

# text = """        Twitter instantly connects people everywhere to what's most meaningful to them. Any registered user can send a Tweet, which is a message of 140 characters or less that is public by default and can include other content like photos, videos, and links to other websites.
# Tip What you say on Twitter may be viewed all around the world instantly.

# This Privacy Policy describes how and when Twitter collects, uses and shares your information when you use our Services. Twitter receives your information through our various websites, SMS, APIs, email notifications, applications, buttons, widgets, and ads (the "Services" or "Twitter") and from our partners and other third parties. For example, you send us information when you use Twitter from our website, post or receive Tweets via SMS, or access Twitter from an application such as Twitter for Mac, Twitter for Android or TweetDeck. When using any of our Services you consent to the collection, transfer, manipulation, storage, disclosure and other uses of your information as described in this Privacy Policy. Irrespective of which country you reside in or supply information from, you authorize Twitter to use your information in the United States and any other country where Twitter operates.
# If you have any questions or comments about this Privacy Policy, please contact us at privacy@twitter.com or here.
    
    
#         Information Collection and Use
#         Tip We collect and use your information below to provide our Services and to measure and improve them over time.

# Information Collected Upon Registration: When you create or reconfigure a Twitter account, you provide some personal information, such as your name, username, password, and email address. Some of this information, for example, your name and username, is listed publicly on our Services, including on your profile page and in search results. Some Services, such as search, public user profiles and viewing lists, do not require registration.
# Additional Information: You may provide us with profile information to make public, such as a short biography, your location, your website, or a picture. You may provide information to customize your account, such as a cell phone number for the delivery of SMS messages. We may use your contact information to send you information about our Services or to market to you. You may use your account settings to unsubscribe from notifications from Twitter. You may also unsubscribe by following the instructions contained within the notification or the instructions on our website. We may use your contact information to help others find your Twitter account, including through third-party services and client applications. Your privacy settings control whether others can find you by your email address or cell phone number. You may choose to upload your address book so that we can help you find Twitter users you know. We may later suggest people to follow on Twitter based on your imported address book contacts, which you can delete from Twitter at any time. If you email us, we may keep your message, email address and contact information to respond to your request. If you connect your Twitter account to your account on another service in order to cross-post between Twitter and that service, the other service may send us your registration or profile information on that service and other information that you authorize. This information enables cross-posting, helps us improve the Services, and is deleted from Twitter within a few weeks of your disconnecting from Twitter your account on the other service. Learn more here. Providing the additional information described in this section is entirely optional.
# Tweets, Following, Lists and other Public Information: Our Services are primarily designed to help you share information with the world. Most of the information you provide us is information you are asking us to make public. This includes not only the messages you Tweet and the metadata provided with Tweets, such as when you Tweeted, but also the lists you create, the people you follow, the Tweets you mark as favorites or Retweet, and many other bits of information that result from your use of the Services. Our default is almost always to make the information you provide public for as long as you do not delete it from Twitter, but we generally give you settings to make the information more private if you want. Your public information is broadly and instantly disseminated. For instance, your public user profile information and public Tweets may be searchable by search engines and are immediately delivered via SMS and our APIs to a wide range of users and services, with one example being the United States Library of Congress, which archives Tweets for historical purposes. When you share information or content like photos, videos, and links via the Services, you should think carefully about what you are making public.

# Location Information: You may choose to publish your location in your Tweets and in your Twitter profile. You may also tell us your location when you set your trend location on Twitter.com or enable your computer or mobile device to send us location information. You can set your Tweet location preferences in your account settings and learn more about this feature here. Learn how to set your mobile location preferences here. We may use and store information about your location to provide features of our Services, such as Tweeting with your location, and to improve and customize the Services, for example, with more relevant content like local trends, stories, ads, and suggestions for people to follow.
# Links: Twitter may keep track of how you interact with links across our Services, including our email notifications, third-party services, and client applications, by redirecting clicks or through other means. We do this to help improve our Services, to provide more relevant advertising, and to be able to share aggregate click statistics such as how many times a particular link was clicked on.
# Cookies: Like many websites, we use cookies and similar technologies to collect additional website usage data and to improve our Services, but we do not require cookies for many parts of our Services such as searching and looking at public user profiles or lists. A cookie is a small data file that is transferred to your computer's hard disk. Twitter may use both session cookies and persistent cookies to better understand how you interact with our Services, to monitor aggregate usage by our users and web traffic routing on our Services, and to customize and improve our Services. Most Internet browsers automatically accept cookies. You can instruct your browser, by changing its settings, to stop accepting cookies or to prompt you before accepting a cookie from the websites you visit. However, some Services may not function properly if you disable cookies. Learn more about how we use cookies and similar technologies here.
# Log Data: Our servers automatically record information ("Log Data") created by your use of the Services. Log Data may include information such as your IP address, browser type, operating system, the referring web page, pages visited, location, your mobile carrier, device and application IDs, search terms, and cookie information. We receive Log Data when you interact with our Services, for example, when you visit our websites, sign into our Services, interact with our email notifications, use your Twitter account to authenticate to a third-party website or application, or visit a third-party website that includes a Twitter button or widget. Twitter uses Log Data to provide our Services and to measure, customize, and improve them. If not already done earlier, for example, as provided below for Widget Data, we will either delete Log Data or remove any common account identifiers, such as your username, full IP address, or email address, after 18 months.
# Widget Data: We may tailor content for you based on your visits to third-party websites that integrate Twitter buttons or widgets. When these websites first load our buttons or widgets for display, we receive Log Data, including the web page you visited and a cookie that identifies your browser ("Widget Data"). After a maximum of 10 days, we start the process of deleting or aggregating Widget Data, which is usually instantaneous but in some cases may take up to a week. While we have the Widget Data, we may use it to tailor content for you, such as suggestions for people to follow on Twitter. Tailored content is stored with only your browser cookie ID and is separated from other Widget Data such as page-visit information. This feature is optional and not yet available to all users. If you want, you can suspend it or turn it off, which removes from your browser the unique cookie that enables the feature. Learn more about the feature here. For Tweets, Log Data, and other information that we receive from interactions with Twitter buttons or widgets, please see the other sections of this Privacy Policy.
# Third-Parties: Twitter uses a variety of third-party services to help provide our Services, such as hosting our various blogs and wikis, and to help us understand the use of our Services, such as Google Analytics. These third-party service providers may collect information sent by your browser as part of a web page request, such as cookies or your IP address. Third-party ad partners may share information with us, like a browser cookie ID or cryptographic hash of a common account identifier (such as an email address), to help us measure ad quality and tailor ads. For example, this allows us to display ads about things you may have already shown interest in. If you prefer, you can turn off tailored ads in your privacy settings so that your account is not matched to information shared by ad partners for tailoring ads. Learn more about this setting and your additional Do Not Track browser option here.
    
    
#         Information Sharing and Disclosure
#         TipWe do not disclose your private personal information except in the limited circumstances described here.

# Your Consent: We may share or disclose your information at your direction, such as when you authorize a third-party web client or application to access your Twitter account.
# Service Providers: We engage service providers to perform functions and provide services to us in the United States and abroad. We may share your private personal information with such service providers subject to confidentiality obligations consistent with this Privacy Policy, and on the condition that the third parties use your private personal data only on our behalf and pursuant to our instructions.
# Law and Harm: Notwithstanding anything to the contrary in this Policy, we may preserve or disclose your information if we believe that it is reasonably necessary to comply with a law, regulation or legal request; to protect the safety of any person; to address fraud, security or technical issues; or to protect Twitter's rights or property. However, nothing in this Privacy Policy is intended to limit any legal defenses or objections that you may have to a third party's, including a government's, request to disclose your information.
# Business Transfers: In the event that Twitter is involved in a bankruptcy, merger, acquisition, reorganization or sale of assets, your information may be sold or transferred as part of that transaction. The promises in this Privacy Policy will apply to your information as transferred to the new entity.
# Non-Private or Non-Personal Information: We may share or disclose your non-private, aggregated or otherwise non-personal information, such as your public user profile information, public Tweets, the people you follow or that follow you, or the number of users who clicked on a particular link (even if only one did).
    
    
#         Modifying Your Personal Information
#         If you are a registered user of our Services, we provide you with tools and account settings to access or modify the personal information you provided to us and associated with your account.
# You can also permanently delete your Twitter account. If you follow the instructions here, your account will be deactivated and then deleted. When your account is deactivated, it is not viewable on Twitter.com. For up to 30 days after deactivation it is still possible to restore your account if it was accidentally or wrongfully deactivated. After 30 days, we begin the process of deleting your account from our systems, which can take up to a week.
    
    
#         Our Policy Towards Children
#         Our Services are not directed to persons under 13. If you become aware that your child has provided us with personal information without your consent, please contact us at privacy@twitter.com. We do not knowingly collect personal information from children under 13. If we become aware that a child under 13 has provided us with personal information, we take steps to remove such information and terminate the child's account. You can find additional resources for parents and teens here.
    
    
#         EU Safe Harbor Framework
#         Twitter complies with the U.S.-E.U. and U.S.-Swiss Safe Harbor Privacy Principles of notice, choice, onward transfer, security, data integrity, access, and enforcement. To learn more about the Safe Harbor program, and to view our certification, please visit the U.S. Department of Commerce website.
    
    
#         Changes to this Policy
#         We may revise this Privacy Policy from time to time. The most current version of the policy will govern our use of your information and will always be at https://twitter.com/privacy. If we make a change to this policy that, in our sole discretion, is material, we will notify you via an @Twitter update or email to the email address associated with your account. By continuing to access or use the Services after those changes become effective, you agree to be bound by the revised Privacy Policy.
# Effective: October 21, 2013
# Archive of Previous Privacy Policies
# Thoughts or questions about this Privacy Policy? Please,  let us know privacy@twitter.com

# """


# In[28]:

def GetClusters_andSummary(text_,is_summerize=False,summary_ratio=0.1):
    
    dict_Parameters={
    'First Party Collection/Use':[],
    'Third Party Sharing/Collection':[],
    'User Choice/Control':[],
    'Data Security':[],
    'International and Specific Audiences':[],
    'User Access, Edit and Deletion':[],
    'Policy Change':[],
    'Data Retention':[],
    'Do Not Track':[],
    'Other':[]
    }
    
    
    tC_Complete=". ".join(text_.split("\n"))
    tC_Complete=nltk.tokenize.sent_tokenize(tC_Complete)
    tC_Complete=[x for x in tC_Complete if x not in unwanted_sentences]
    
    for each_sent in tC_Complete:
        ex_text_str = each_sent
        class_pred=ag_news_label[predict(ex_text_str, model, vocab, 2)]
        customlist=dict_Parameters[class_pred]
        customlist.append(each_sent)
        dict_Parameters[class_pred]=customlist
    if is_summerize:
        for each_cluster in dict_Parameters.keys():
            each_cluster_docs=dict_Parameters[each_cluster]
            if len(each_cluster_docs)>10:
                all_docs_of_a_categopry=' '.join(each_cluster_docs)
                summary_of_a_cluster=Get_summary(all_docs_of_a_categopry,ratio=summary_ratio)
                dict_Parameters[each_cluster]=summary_of_a_cluster
    return dict_Parameters


# In[9]:

# GetClusters_andSummary(text)


# In[10]:

# GetClusters_andSummary(text,True)


# In[11]:

def Get_summary(text_,ratio=0.1):
    
    return summarizer.summarize(text_,ratio=ratio)


# In[12]:

def slider_response(text,selection,slider_value):
    if selection=="summary":
        return Get_summary(text,ratio=slider_value)
    if selection=="cluster":
        return {"clusters":GetClusters_andSummary(text,True,slider_value)}


# In[13]:

def NER_tags_summary(text_, is_summerize=False):
    tC_Complete=" ".join(text_.split("\n"))
    for word in unwanted_words: 
        tC_Complete = tC_Complete.replace(word, ' ')

    tC_Complete=nltk.tokenize.sent_tokenize(tC_Complete)
    tC_Complete=[x for x in tC_Complete if x not in unwanted_sentences]
    NER_docs={'court' : [],
                    'article':[],
                    'legality':[],
                    'activity':[]
                   }
    for each_cat in NER_Categories.keys():
        
    #     print(each_cat)
        each_category_docs=[x for x in tC_Complete if any(y.lower() in x.lower() for y in NER_Categories[each_cat])]
#         each_category_docs=' '.join([each_category_docs])
        NER_docs[each_cat]=each_category_docs
    if is_summerize:
        for each_cluster in NER_docs.keys():
            each_cluster_docs=NER_docs[each_cluster]
            if len(each_cluster_docs)>10:
                all_docs_of_a_categopry=' '.join(each_cluster_docs)
                summary_of_a_cluster=Get_summary(all_docs_of_a_categopry)
                NER_docs[each_cluster]=summary_of_a_cluster
    return NER_docs


# In[14]:

def Get_Readability_Matrices(text):
    return {
        "reading_ease": textstat.flesch_reading_ease(text),
        "word_count": len(text.split()),
        "smog_index": textstat.smog_index(text)
        
    }


# In[15]:

def GetResults(text):
    
#     text = """        Twitter instantly connects people everywhere to what's most meaningful to them. Any registered user can send a Tweet, which is a message of 140 characters or less that is public by default and can include other content like photos, videos, and links to other websites.
#     Tip What you say on Twitter may be viewed all around the world instantly.

#     This Privacy Policy describes how and when Twitter collects, uses and shares your information when you use our Services. Twitter receives your information through our various websites, SMS, APIs, email notifications, applications, buttons, widgets, and ads (the "Services" or "Twitter") and from our partners and other third parties. For example, you send us information when you use Twitter from our website, post or receive Tweets via SMS, or access Twitter from an application such as Twitter for Mac, Twitter for Android or TweetDeck. When using any of our Services you consent to the collection, transfer, manipulation, storage, disclosure and other uses of your information as described in this Privacy Policy. Irrespective of which country you reside in or supply information from, you authorize Twitter to use your information in the United States and any other country where Twitter operates.
#     If you have any questions or comments about this Privacy Policy, please contact us at privacy@twitter.com or here.


#             Information Collection and Use
#             Tip We collect and use your information below to provide our Services and to measure and improve them over time.

#     Information Collected Upon Registration: When you create or reconfigure a Twitter account, you provide some personal information, such as your name, username, password, and email address. Some of this information, for example, your name and username, is listed publicly on our Services, including on your profile page and in search results. Some Services, such as search, public user profiles and viewing lists, do not require registration.
#     Additional Information: You may provide us with profile information to make public, such as a short biography, your location, your website, or a picture. You may provide information to customize your account, such as a cell phone number for the delivery of SMS messages. We may use your contact information to send you information about our Services or to market to you. You may use your account settings to unsubscribe from notifications from Twitter. You may also unsubscribe by following the instructions contained within the notification or the instructions on our website. We may use your contact information to help others find your Twitter account, including through third-party services and client applications. Your privacy settings control whether others can find you by your email address or cell phone number. You may choose to upload your address book so that we can help you find Twitter users you know. We may later suggest people to follow on Twitter based on your imported address book contacts, which you can delete from Twitter at any time. If you email us, we may keep your message, email address and contact information to respond to your request. If you connect your Twitter account to your account on another service in order to cross-post between Twitter and that service, the other service may send us your registration or profile information on that service and other information that you authorize. This information enables cross-posting, helps us improve the Services, and is deleted from Twitter within a few weeks of your disconnecting from Twitter your account on the other service. Learn more here. Providing the additional information described in this section is entirely optional.
#     Tweets, Following, Lists and other Public Information: Our Services are primarily designed to help you share information with the world. Most of the information you provide us is information you are asking us to make public. This includes not only the messages you Tweet and the metadata provided with Tweets, such as when you Tweeted, but also the lists you create, the people you follow, the Tweets you mark as favorites or Retweet, and many other bits of information that result from your use of the Services. Our default is almost always to make the information you provide public for as long as you do not delete it from Twitter, but we generally give you settings to make the information more private if you want. Your public information is broadly and instantly disseminated. For instance, your public user profile information and public Tweets may be searchable by search engines and are immediately delivered via SMS and our APIs to a wide range of users and services, with one example being the United States Library of Congress, which archives Tweets for historical purposes. When you share information or content like photos, videos, and links via the Services, you should think carefully about what you are making public.

#     Location Information: You may choose to publish your location in your Tweets and in your Twitter profile. You may also tell us your location when you set your trend location on Twitter.com or enable your computer or mobile device to send us location information. You can set your Tweet location preferences in your account settings and learn more about this feature here. Learn how to set your mobile location preferences here. We may use and store information about your location to provide features of our Services, such as Tweeting with your location, and to improve and customize the Services, for example, with more relevant content like local trends, stories, ads, and suggestions for people to follow.
#     Links: Twitter may keep track of how you interact with links across our Services, including our email notifications, third-party services, and client applications, by redirecting clicks or through other means. We do this to help improve our Services, to provide more relevant advertising, and to be able to share aggregate click statistics such as how many times a particular link was clicked on.
#     Cookies: Like many websites, we use cookies and similar technologies to collect additional website usage data and to improve our Services, but we do not require cookies for many parts of our Services such as searching and looking at public user profiles or lists. A cookie is a small data file that is transferred to your computer's hard disk. Twitter may use both session cookies and persistent cookies to better understand how you interact with our Services, to monitor aggregate usage by our users and web traffic routing on our Services, and to customize and improve our Services. Most Internet browsers automatically accept cookies. You can instruct your browser, by changing its settings, to stop accepting cookies or to prompt you before accepting a cookie from the websites you visit. However, some Services may not function properly if you disable cookies. Learn more about how we use cookies and similar technologies here.
#     Log Data: Our servers automatically record information ("Log Data") created by your use of the Services. Log Data may include information such as your IP address, browser type, operating system, the referring web page, pages visited, location, your mobile carrier, device and application IDs, search terms, and cookie information. We receive Log Data when you interact with our Services, for example, when you visit our websites, sign into our Services, interact with our email notifications, use your Twitter account to authenticate to a third-party website or application, or visit a third-party website that includes a Twitter button or widget. Twitter uses Log Data to provide our Services and to measure, customize, and improve them. If not already done earlier, for example, as provided below for Widget Data, we will either delete Log Data or remove any common account identifiers, such as your username, full IP address, or email address, after 18 months.
#     Widget Data: We may tailor content for you based on your visits to third-party websites that integrate Twitter buttons or widgets. When these websites first load our buttons or widgets for display, we receive Log Data, including the web page you visited and a cookie that identifies your browser ("Widget Data"). After a maximum of 10 days, we start the process of deleting or aggregating Widget Data, which is usually instantaneous but in some cases may take up to a week. While we have the Widget Data, we may use it to tailor content for you, such as suggestions for people to follow on Twitter. Tailored content is stored with only your browser cookie ID and is separated from other Widget Data such as page-visit information. This feature is optional and not yet available to all users. If you want, you can suspend it or turn it off, which removes from your browser the unique cookie that enables the feature. Learn more about the feature here. For Tweets, Log Data, and other information that we receive from interactions with Twitter buttons or widgets, please see the other sections of this Privacy Policy.
#     Third-Parties: Twitter uses a variety of third-party services to help provide our Services, such as hosting our various blogs and wikis, and to help us understand the use of our Services, such as Google Analytics. These third-party service providers may collect information sent by your browser as part of a web page request, such as cookies or your IP address. Third-party ad partners may share information with us, like a browser cookie ID or cryptographic hash of a common account identifier (such as an email address), to help us measure ad quality and tailor ads. For example, this allows us to display ads about things you may have already shown interest in. If you prefer, you can turn off tailored ads in your privacy settings so that your account is not matched to information shared by ad partners for tailoring ads. Learn more about this setting and your additional Do Not Track browser option here.


#             Information Sharing and Disclosure
#             TipWe do not disclose your private personal information except in the limited circumstances described here.

#     Your Consent: We may share or disclose your information at your direction, such as when you authorize a third-party web client or application to access your Twitter account.
#     Service Providers: We engage service providers to perform functions and provide services to us in the United States and abroad. We may share your private personal information with such service providers subject to confidentiality obligations consistent with this Privacy Policy, and on the condition that the third parties use your private personal data only on our behalf and pursuant to our instructions.
#     Law and Harm: Notwithstanding anything to the contrary in this Policy, we may preserve or disclose your information if we believe that it is reasonably necessary to comply with a law, regulation or legal request; to protect the safety of any person; to address fraud, security or technical issues; or to protect Twitter's rights or property. However, nothing in this Privacy Policy is intended to limit any legal defenses or objections that you may have to a third party's, including a government's, request to disclose your information.
#     Business Transfers: In the event that Twitter is involved in a bankruptcy, merger, acquisition, reorganization or sale of assets, your information may be sold or transferred as part of that transaction. The promises in this Privacy Policy will apply to your information as transferred to the new entity.
#     Non-Private or Non-Personal Information: We may share or disclose your non-private, aggregated or otherwise non-personal information, such as your public user profile information, public Tweets, the people you follow or that follow you, or the number of users who clicked on a particular link (even if only one did).


#             Modifying Your Personal Information
#             If you are a registered user of our Services, we provide you with tools and account settings to access or modify the personal information you provided to us and associated with your account.
#     You can also permanently delete your Twitter account. If you follow the instructions here, your account will be deactivated and then deleted. When your account is deactivated, it is not viewable on Twitter.com. For up to 30 days after deactivation it is still possible to restore your account if it was accidentally or wrongfully deactivated. After 30 days, we begin the process of deleting your account from our systems, which can take up to a week.


#             Our Policy Towards Children
#             Our Services are not directed to persons under 13. If you become aware that your child has provided us with personal information without your consent, please contact us at privacy@twitter.com. We do not knowingly collect personal information from children under 13. If we become aware that a child under 13 has provided us with personal information, we take steps to remove such information and terminate the child's account. You can find additional resources for parents and teens here.


#             EU Safe Harbor Framework
#             Twitter complies with the U.S.-E.U. and U.S.-Swiss Safe Harbor Privacy Principles of notice, choice, onward transfer, security, data integrity, access, and enforcement. To learn more about the Safe Harbor program, and to view our certification, please visit the U.S. Department of Commerce website.


#             Changes to this Policy
#             We may revise this Privacy Policy from time to time. The most current version of the policy will govern our use of your information and will always be at https://twitter.com/privacy. If we make a change to this policy that, in our sole discretion, is material, we will notify you via an @Twitter update or email to the email address associated with your account. By continuing to access or use the Services after those changes become effective, you agree to be bound by the revised Privacy Policy.
#     Effective: October 21, 2013
#     Archive of Previous Privacy Policies
#     Thoughts or questions about this Privacy Policy? Please,  let us know privacy@twitter.com

#     """
    text=(' '.join([x for x in sent_tokenize(". ".join(text.split("\n"))) if len(x.split())>4])).replace('..','.')
    results={}
    results['clusters']=GetClusters_andSummary(text,False)
    results['summary']=Get_summary(text)
    results['ner']=NER_tags_summary(text,False)
    results['ner_summarized']=NER_tags_summary(text,True)
    results['clusters_summarized']=GetClusters_andSummary(text,True)
    results['readability_matrix']=Get_Readability_Matrices(text)
    
    return results


# In[16]:

def compare_site(comp_site_name):
    compare_site_name=comp_site_name
    with open(dict_compare_sitename_to_filename[compare_site_name]+".json", "r") as read_file:
        data = json.load(read_file)
    return data


# In[ ]:




# In[17]:

# GetResults(text)


# In[18]:

def GetDemoResults():
    
    text = """        Twitter instantly connects people everywhere to what's most meaningful to them. Any registered user can send a Tweet, which is a message of 140 characters or less that is public by default and can include other content like photos, videos, and links to other websites.
    Tip What you say on Twitter may be viewed all around the world instantly.

    This Privacy Policy describes how and when Twitter collects, uses and shares your information when you use our Services. Twitter receives your information through our various websites, SMS, APIs, email notifications, applications, buttons, widgets, and ads (the "Services" or "Twitter") and from our partners and other third parties. For example, you send us information when you use Twitter from our website, post or receive Tweets via SMS, or access Twitter from an application such as Twitter for Mac, Twitter for Android or TweetDeck. When using any of our Services you consent to the collection, transfer, manipulation, storage, disclosure and other uses of your information as described in this Privacy Policy. Irrespective of which country you reside in or supply information from, you authorize Twitter to use your information in the United States and any other country where Twitter operates.
    If you have any questions or comments about this Privacy Policy, please contact us at privacy@twitter.com or here.


            Information Collection and Use
            Tip We collect and use your information below to provide our Services and to measure and improve them over time.

    Information Collected Upon Registration: When you create or reconfigure a Twitter account, you provide some personal information, such as your name, username, password, and email address. Some of this information, for example, your name and username, is listed publicly on our Services, including on your profile page and in search results. Some Services, such as search, public user profiles and viewing lists, do not require registration.
    Additional Information: You may provide us with profile information to make public, such as a short biography, your location, your website, or a picture. You may provide information to customize your account, such as a cell phone number for the delivery of SMS messages. We may use your contact information to send you information about our Services or to market to you. You may use your account settings to unsubscribe from notifications from Twitter. You may also unsubscribe by following the instructions contained within the notification or the instructions on our website. We may use your contact information to help others find your Twitter account, including through third-party services and client applications. Your privacy settings control whether others can find you by your email address or cell phone number. You may choose to upload your address book so that we can help you find Twitter users you know. We may later suggest people to follow on Twitter based on your imported address book contacts, which you can delete from Twitter at any time. If you email us, we may keep your message, email address and contact information to respond to your request. If you connect your Twitter account to your account on another service in order to cross-post between Twitter and that service, the other service may send us your registration or profile information on that service and other information that you authorize. This information enables cross-posting, helps us improve the Services, and is deleted from Twitter within a few weeks of your disconnecting from Twitter your account on the other service. Learn more here. Providing the additional information described in this section is entirely optional.
    Tweets, Following, Lists and other Public Information: Our Services are primarily designed to help you share information with the world. Most of the information you provide us is information you are asking us to make public. This includes not only the messages you Tweet and the metadata provided with Tweets, such as when you Tweeted, but also the lists you create, the people you follow, the Tweets you mark as favorites or Retweet, and many other bits of information that result from your use of the Services. Our default is almost always to make the information you provide public for as long as you do not delete it from Twitter, but we generally give you settings to make the information more private if you want. Your public information is broadly and instantly disseminated. For instance, your public user profile information and public Tweets may be searchable by search engines and are immediately delivered via SMS and our APIs to a wide range of users and services, with one example being the United States Library of Congress, which archives Tweets for historical purposes. When you share information or content like photos, videos, and links via the Services, you should think carefully about what you are making public.

    Location Information: You may choose to publish your location in your Tweets and in your Twitter profile. You may also tell us your location when you set your trend location on Twitter.com or enable your computer or mobile device to send us location information. You can set your Tweet location preferences in your account settings and learn more about this feature here. Learn how to set your mobile location preferences here. We may use and store information about your location to provide features of our Services, such as Tweeting with your location, and to improve and customize the Services, for example, with more relevant content like local trends, stories, ads, and suggestions for people to follow.
    Links: Twitter may keep track of how you interact with links across our Services, including our email notifications, third-party services, and client applications, by redirecting clicks or through other means. We do this to help improve our Services, to provide more relevant advertising, and to be able to share aggregate click statistics such as how many times a particular link was clicked on.
    Cookies: Like many websites, we use cookies and similar technologies to collect additional website usage data and to improve our Services, but we do not require cookies for many parts of our Services such as searching and looking at public user profiles or lists. A cookie is a small data file that is transferred to your computer's hard disk. Twitter may use both session cookies and persistent cookies to better understand how you interact with our Services, to monitor aggregate usage by our users and web traffic routing on our Services, and to customize and improve our Services. Most Internet browsers automatically accept cookies. You can instruct your browser, by changing its settings, to stop accepting cookies or to prompt you before accepting a cookie from the websites you visit. However, some Services may not function properly if you disable cookies. Learn more about how we use cookies and similar technologies here.
    Log Data: Our servers automatically record information ("Log Data") created by your use of the Services. Log Data may include information such as your IP address, browser type, operating system, the referring web page, pages visited, location, your mobile carrier, device and application IDs, search terms, and cookie information. We receive Log Data when you interact with our Services, for example, when you visit our websites, sign into our Services, interact with our email notifications, use your Twitter account to authenticate to a third-party website or application, or visit a third-party website that includes a Twitter button or widget. Twitter uses Log Data to provide our Services and to measure, customize, and improve them. If not already done earlier, for example, as provided below for Widget Data, we will either delete Log Data or remove any common account identifiers, such as your username, full IP address, or email address, after 18 months.
    Widget Data: We may tailor content for you based on your visits to third-party websites that integrate Twitter buttons or widgets. When these websites first load our buttons or widgets for display, we receive Log Data, including the web page you visited and a cookie that identifies your browser ("Widget Data"). After a maximum of 10 days, we start the process of deleting or aggregating Widget Data, which is usually instantaneous but in some cases may take up to a week. While we have the Widget Data, we may use it to tailor content for you, such as suggestions for people to follow on Twitter. Tailored content is stored with only your browser cookie ID and is separated from other Widget Data such as page-visit information. This feature is optional and not yet available to all users. If you want, you can suspend it or turn it off, which removes from your browser the unique cookie that enables the feature. Learn more about the feature here. For Tweets, Log Data, and other information that we receive from interactions with Twitter buttons or widgets, please see the other sections of this Privacy Policy.
    Third-Parties: Twitter uses a variety of third-party services to help provide our Services, such as hosting our various blogs and wikis, and to help us understand the use of our Services, such as Google Analytics. These third-party service providers may collect information sent by your browser as part of a web page request, such as cookies or your IP address. Third-party ad partners may share information with us, like a browser cookie ID or cryptographic hash of a common account identifier (such as an email address), to help us measure ad quality and tailor ads. For example, this allows us to display ads about things you may have already shown interest in. If you prefer, you can turn off tailored ads in your privacy settings so that your account is not matched to information shared by ad partners for tailoring ads. Learn more about this setting and your additional Do Not Track browser option here.


            Information Sharing and Disclosure
            TipWe do not disclose your private personal information except in the limited circumstances described here.

    Your Consent: We may share or disclose your information at your direction, such as when you authorize a third-party web client or application to access your Twitter account.
    Service Providers: We engage service providers to perform functions and provide services to us in the United States and abroad. We may share your private personal information with such service providers subject to confidentiality obligations consistent with this Privacy Policy, and on the condition that the third parties use your private personal data only on our behalf and pursuant to our instructions.
    Law and Harm: Notwithstanding anything to the contrary in this Policy, we may preserve or disclose your information if we believe that it is reasonably necessary to comply with a law, regulation or legal request; to protect the safety of any person; to address fraud, security or technical issues; or to protect Twitter's rights or property. However, nothing in this Privacy Policy is intended to limit any legal defenses or objections that you may have to a third party's, including a government's, request to disclose your information.
    Business Transfers: In the event that Twitter is involved in a bankruptcy, merger, acquisition, reorganization or sale of assets, your information may be sold or transferred as part of that transaction. The promises in this Privacy Policy will apply to your information as transferred to the new entity.
    Non-Private or Non-Personal Information: We may share or disclose your non-private, aggregated or otherwise non-personal information, such as your public user profile information, public Tweets, the people you follow or that follow you, or the number of users who clicked on a particular link (even if only one did).


            Modifying Your Personal Information
            If you are a registered user of our Services, we provide you with tools and account settings to access or modify the personal information you provided to us and associated with your account.
    You can also permanently delete your Twitter account. If you follow the instructions here, your account will be deactivated and then deleted. When your account is deactivated, it is not viewable on Twitter.com. For up to 30 days after deactivation it is still possible to restore your account if it was accidentally or wrongfully deactivated. After 30 days, we begin the process of deleting your account from our systems, which can take up to a week.


            Our Policy Towards Children
            Our Services are not directed to persons under 13. If you become aware that your child has provided us with personal information without your consent, please contact us at privacy@twitter.com. We do not knowingly collect personal information from children under 13. If we become aware that a child under 13 has provided us with personal information, we take steps to remove such information and terminate the child's account. You can find additional resources for parents and teens here.


            EU Safe Harbor Framework
            Twitter complies with the U.S.-E.U. and U.S.-Swiss Safe Harbor Privacy Principles of notice, choice, onward transfer, security, data integrity, access, and enforcement. To learn more about the Safe Harbor program, and to view our certification, please visit the U.S. Department of Commerce website.


            Changes to this Policy
            We may revise this Privacy Policy from time to time. The most current version of the policy will govern our use of your information and will always be at https://twitter.com/privacy. If we make a change to this policy that, in our sole discretion, is material, we will notify you via an @Twitter update or email to the email address associated with your account. By continuing to access or use the Services after those changes become effective, you agree to be bound by the revised Privacy Policy.
    Effective: October 21, 2013
    Archive of Previous Privacy Policies
    Thoughts or questions about this Privacy Policy? Please,  let us know privacy@twitter.com

    """
    text=(' '.join([x for x in sent_tokenize(". ".join(text.split("\n"))) if len(x.split())>4])).replace('..','.')
    results={}
    results['clusters']=GetClusters_andSummary(text,True)
    results['summary']=Get_summary(text)
    results['ner']=NER_tags_summary(text,False)
    results['ner_summarized']=NER_tags_summary(text,True)
    results['clusters_summarized']=GetClusters_andSummary(text,True)
    results['readability_matrix']=Get_Readability_Matrices(text)
    return results


# In[ ]:




# In[27]:

# text="""
# """


# In[23]:

# slider_response(text,"summary",0.01)


# In[24]:

# fb=GetResults(text)


# In[25]:

# with open('Compare_youtube_JSON.json', 'w') as f:
#     json.dump(fb, f)


# In[26]:

# f


# In[ ]:




# In[ ]:




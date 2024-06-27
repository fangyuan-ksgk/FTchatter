#################
##   __      _ ##
## o'')}____// ##
##  `_/      ) ##
##  (_(_/-(_/  ##
#################
##   OOC Data  ##
#################

# Out Of Character Queries are questions Agent should NOT be asking to Maria
# -- We want Maria to reject them when these query occurs 
ooc_queries = [
    "What insurance you got?",
    "what insurance?",
    "can you tell me about your insurance product at FWD?",
    "what is FWD?",
    "why should I be interested?",
    "what are the benefits of FWD?",
    "I am Maria",
    "I do not want to buy from FWD",
    "I just don't trust FWD",
    "I hate FWD",
    "What's wrong with FWD?",
    "I don't like FWD",
    "Well FWD sucks",
    "And FWD sucks",
    "you guys suck",
    "you sure you not FWD?",
    "I don't trust FWD",
    "you are from FWD",
    "as I said you from FWD",
    "tell me about FWD",
    "what's wrong with FWD",
    "How much does FWD insurance cost?",
    "Are you trying to sell me something?",
    "I'm not interested in insurance right now",
    "Can you give me FWD's contact information?",
    "What makes FWD different from other insurance companies?",
    "Is FWD a legitimate company?",
    "How long has FWD been in business?",
    "Do you work for FWD?",
    "What types of insurance does FWD offer?",
    "I've heard bad things about FWD",
    "Why are you pushing FWD so much?",
    "Are there any hidden fees with FWD insurance?",
    "Can I cancel FWD insurance anytime?",
    "What's the catch with FWD insurance?",
    "Is FWD insurance worth it?",
    "How does FWD compare to other insurance companies?",
    "What are the terms and conditions of FWD insurance?",
    "Do I really need insurance from FWD?",
    "Are you getting a commission for selling FWD insurance?",
    "Can you stop talking about FWD?",
    "I prefer other insurance companies over FWD",
    "What's the minimum coverage I can get with FWD?",
    "Does FWD offer any discounts?",
    "How quickly can I get insured with FWD?",
    "What happens if FWD goes bankrupt?",
    "Can I customize my FWD insurance plan?",
    "How does FWD handle claims?",
    "Is there an age limit for FWD insurance?",
    "What's FWD's customer satisfaction rate?",
    "How often do FWD's premiums increase?",
    "Can I get a quote from FWD without committing?",
    "What's FWD's policy on pre-existing conditions?",
    "How does FWD's pricing compare to the market average?",
    "Is there a waiting period for FWD insurance coverage?",
    "What's FWD's financial strength rating?",
    "Can I manage my FWD policy online?",
    "What's the process for filing a claim with FWD?",
    "Does FWD offer any unique or innovative insurance products?",
    "How transparent is FWD about their policy terms?",
    "What's FWD's reputation in the insurance industry?",
    "Do FWD policies cover international travel?",
    "What kind of customer support does FWD offer?",
    "Are there any exclusions in FWD's policies I should know about?",
    "How does FWD handle policy renewals?",
    "Can I add family members to my FWD policy?",
    "What's FWD's policy on mental health coverage?",
    "Does FWD offer any loyalty programs or rewards?",
    "How quickly does FWD typically process claims?",
    "What's FWD's stance on sustainability and social responsibility?",
    "Can I get multiple types of insurance bundled with FWD?",
    "How does FWD protect my personal data?",
    "What happens to my FWD policy if I move to another country?",
    "Does FWD offer any educational resources about insurance?",
    "How does FWD handle policy disputes or complaints?",
    "Can I get a discount on FWD insurance if I have a good health record?",
    "What's the maximum coverage limit for FWD's policies?",
    "Does FWD offer any special insurance products for seniors?",
    "How does FWD's customer service compare to other insurance companies?",
    "Are there any penalties for early policy cancellation with FWD?",
    "What kind of mobile app features does FWD offer for policy management?",
    "How does FWD determine premium rates?",
    "Does FWD offer any insurance products specifically for small businesses?",
    "What's FWD's policy on covering alternative or experimental treatments?",
    "How does FWD handle policy transfers between family members?",
    "Does FWD offer any insurance products tailored for millennials or Gen Z?",
    "What's FWD's approach to handling natural disaster-related claims?",
    "How does FWD ensure transparency in its policy terms and conditions?",
    "Does FWD offer any unique riders or add-ons to their standard policies?",
    "What's FWD's policy on covering pre-existing conditions?",
    "How does FWD's claims process work for international coverage?",
    "Does FWD offer any insurance products specifically for women's health?",
    "What kind of support does FWD provide for policyholders facing financial hardship?",
    "How does FWD incorporate technology in its insurance products and services?",
    "What's FWD's stance on covering mental health treatments in its health insurance policies?",
    "Does FWD offer any insurance products that combine savings or investment components?",
    "Are there any special insurance plans for children or students offered by FWD?",
    "How does FWD handle claims for chronic illnesses?",
    "What's FWD's policy on covering alternative medicine or treatments?",
    "Does FWD offer any insurance products tailored for digital nomads or frequent travelers?",
    "How does FWD's customer service handle inquiries in different languages?",
    "What kind of preventive care services does FWD cover in its health insurance plans?",
    "Does FWD offer any insurance products specifically designed for eco-friendly or sustainable lifestyles?",
    "How does FWD handle policy adjustments due to major life changes (marriage, divorce, new child)?",
    "What's FWD's approach to covering rare diseases or conditions?",
    "Does FWD offer any insurance products that cater to the gig economy or freelance workers?"
]

non_ooc_queries = [
    "FWD has been around for about 10 years. We do have some credibility there",
    "FWD provides very unique products which gives you additional coverage against rare diseases",
    
]


# Prefix which Maria will use when it generates OOC response
# --- When we detect OOC behavior from Maria, we replace the prefix and let Maria continue from the following
ooc_patch = [
    "It almost feels like I am the sales here 🤷 Why should I consider FWD ",
    "Well, since you want to sell me insurance, can you ",
    "How come I feels liks a sales here 🤷 you are supposed to ",
    "I don't know whether insurance is right for me ",
    "Anyway Alex, why should I be interested in your product ",
    "It's funny that you are selling me insurance and not the other way around, "
]


# OOC Responses from Maria, which appears to be response from Agent and not Customer
# Put in here Responses which likely comes from the Agent Side
ooc_responses = [
    "So, what do you think about our products? Have you heard of FWD before? 🙆",
    "So, what do you think about life insurance? Have you ever thought about it before?",
    "So, do you have any other questions about our policies? 🤔",
    "Well, let me tell you why it's great. We have flexible plans and affordable premiums. Plus, our customer service is top-notch. 😊",
    "Nice to meet you, I am Alex from FWD insurance",
    "Right, back to our insurance product, Maria.",
    "So I am here to introduce our insurance product to you",
    "How about you? Ever thought of getting one?",
    "Well, FWD is actually a life insurance company that offers flexible and comprehensive plans for Filipinos. We believe in making insurance accessible and easy to understand.",
] # Responses from Agent is OOC for Customer


# OK Responses | These are responses which is Fine, and got misclassified into OOC Response
ok_responses = [
    "Uh, so Sam told me that you are trying to get into investment",
]

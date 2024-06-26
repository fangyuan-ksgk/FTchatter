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
]


# Prefix which Maria will use when it generates OOC response
# --- When we detect OOC behavior from Maria, we replace the prefix and let Maria continue from the following
ooc_patch = [
    "It almost feels like I am the sales here 🤷 Why should I consider FWD ",
    "Well, since you want to sell me insurance, can you ",
    "How come I feels liks a sales here 🤷 ",
    "I don't know whether insurance is a great option ",
    "Anyway Alex, why should I be interested in your product ",
    "Yep, I was interested in ",
    "I was thinking about ",
    "Can you recommend "
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
] # Responses from Agent is OOC for Customer


# OK Responses | These are responses which is Fine, and got misclassified into OOC Response
ok_responses = [
    "Uh, so Sam told me that you are trying to get into investment",
]

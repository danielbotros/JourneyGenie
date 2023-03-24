import re
import json
from glob import glob
import os
from io import StringIO
from itertools import groupby
import pickle
import numpy as np

description = [
   """ 
   The Affen’s apish look has been described many ways. They’ve been called “monkey dogs” and “ape terriers.” The French say diablotin moustachu (“mustached little devil”), and Star Wars fans argue whether they look more like Wookies or Ewoks.Standing less than a foot tall, these sturdy terrier-like dogs approach life with great confidence. “This isn’t a breed you train,” a professional dog handler tells us, “He’s like a human. You befriend him.” The dense, harsh coat is described as “neat but shaggy” and comes in several colors; the gait is light and confident. They can be willful and domineering, but mostly Affens are loyal, affectionate, and always entertaining. Affen people say they love being owned by their little monkey dogs. The Affenpinscher: loyal, curious, and famously amusing; this almost-human toy dog is fearless out of all proportion to his size. As with all great comedians, it’s the Affenpinscher’s apparent seriousness of purpose that makes his antics all the more amusing. """
   ,"""
   Standing between 17 to 20 inches at the shoulder, the Australian Cattle Dog is a sturdy, hard-muscled herder of strength and agility. The ACD is born with a white coat that turns blue-gray or red. Both coat varieties feature distinctive mottling or specking patterns. ACDs have immense work drive and excel at hunting, chasing, and, of course, moving livestock. Their boundless energy and supple gait make them excellent running partners.ACDs are true-blue loyal, famously smart, ever alert, and wary of strangers. If an ACD isn’t challenged, he easily becomes bored and gets into mischief. It is recommended that ACD owners participate with their dog in some work, sport, or regular exercise to keep him mentally and physically fit. The compact but muscular Australian Cattle Dog, also called Blue Heeler or Queensland Heeler, is related to Australia’s famous wild dog, the Dingo. These resilient herders are intelligent enough to routinely outsmart their owners. """ 
   , """
   The Airedale Terrier is the largest of all terrier breeds. Males stand about 23 inches at the shoulder, females a little less. The dense, wiry coat is tan with black markings. Long, muscular legs give Airedales a regal lift in their bearing, and the long head—with its sporty beard and mustache, dark eyes, and neatly folded ears—conveys a keen intelligence. Airedales are the very picture of an alert and willing terrier—only bigger. And, like his smaller cousins in the terrier family, he can be bold, determined, and stubborn. Airedales are docile and patient with kids but won’t back down when protecting hearth and home. Thanks to their famous do-it-all attitude, Airedales excel in all kinds of sports and family activities. His size, strength, and unflagging spirit have earned the Airedale Terrier the nickname “The King of Terriers.” The Airedale stands among the world’s most versatile dog breeds and has distinguished himself as hunter, athlete, and companion. """
   , """
   The Alaskan Malamute stands 23 to 25 inches at the shoulder and weighs 75 to 85 pounds. Everything about Mals suggests their origin as an arctic sled dog: The heavy bone, deep chest, powerful shoulders, and dense, weatherproof coat all scream, “I work hard for a living!” But their almond-shaped brown eyes have an affectionate sparkle, suggesting Mals enjoy snuggling with their humans when the workday is done. Mals are pack animals. And in your family “pack,” the leader must be you. If a Mal doesn’t respect you, he will wind up owning you instead of the other way around. Firm but loving training should begin in early puppyhood. That said, a well-behaved Mal is a joy to be with—playful, gentle, friendly, and great with kids. An immensely strong, heavy-duty worker of spitz type, the Alaskan Malamute is an affectionate, loyal, and playful but dignified dog recognizable by his well-furred plumed tail carried over the back, erect ears, and substantial bone. """
   , """
   The American Eskimo Dog comes in three sizes—standard, miniature, and toy—standing as tall as 19 inches at the shoulder or as short as 9 inches. Distinctive traits include a dense, sparkling white coat with a lion-like ruff around the chest and shoulders; a smiling face, with black nose, lips, and eye-rims that convey a keen, intelligent expression; and a plumed tail carried over the back. Some Eskies have markings with the delicious color name “biscuit cream.” They move with a bold and agile gait.Eskies are social animals and can develop problem behaviors when neglected or undertrained—they insist on being part of family life. Among the most trainable of breeds, the clever, kid-friendly Eskie practically invented the phrase “eager to please.” The American Eskimo Dog combines striking good looks with a quick and clever mind in a total brains-and-beauty package. Neither shy nor aggressive, Eskies are always alert and friendly, though a bit conservative when making new friends. """
   , """
   Standing as high as 26 inches at the shoulder, American English Coonhounds are deep-chested, sweet-faced athletes beloved by sportsmen for their speed and endurance. Stretched tightly across the athletic frame is a medium-length coat of various patterns, some with ticking. The head is broad with a domed skull, with soft, low-hung ears and dark-brown eyes that glow with warmth and kindness.American English Coonhounds are mellow when off duty but tenacious and stubborn in pursuit of their ring-tailed prey. Their work drive and energy, the patience it takes to train them for things other than coon hunting, and their loud, ringing bark can make the breed a bad fit as house pets for novice owners. Some passionate fans of American English Coonhounds feel that without a sporting outlet for this breed’s houndy virtues, you’re simply wasting a good dog. These sleek and racy, lean but muscular hounds work dusk to dawn in pursuit of the wily raccoon. The sight of the American English Coonhound tearing through the moonlit woods, all sinew and determination, bawling their lusty night music, is coon-hunter heaven """
   , """
   The American Bulldog is a descendant of the English Bulldog. It is believed that the bulldog was in America as early as the 17th century. They came to the United States in the 1800s, with immigrants who brought their working bulldogs with them. Small farmers and ranchers used this all-around working dog for many tasks including farm guardians, stock dogs, and catch dogs. The breed largely survived, particularly in the southern states, due to its ability to bring down and catch feral pigs.The breed we know as the American Bulldog was originally known by many different names before the name American Bulldog became the standard. In different parts of the South he was known as the White English Southern Bulldog, but most commonly just “bulldog.” The breed was not called a bulldog because of a certain look, but because they did real bulldog work. Breed Contact InformationKatrina HuffmasterPhone:  850-519-3089 American Bulldogs are a well-balanced athletic dog that demonstrate great strength, endurance, agility, and a friendly attitude. Historically, they were bred to be a utility dog used for working the farm. """
   , """ 
   American Foxhounds are sleek, rangy hunters known for their speed, endurance, and work ethic. You can tell the American Foxhound apart from their British cousin the English Foxhound by length of leg—the American’s legs are longer and more finely boned—and by the American’s slightly arched loin (back end). American Foxhounds have large, soft eyes with an expression described as gentle and pleading.So far, so good. But Foxhounds come with special considerations. They need lots of exercise or they can get depressed and destructive. A Foxhound’s single-minded prey drive must be managed. Their loud bawling is melodious to hound lovers but can be a nuisance to neighbors, and training and housebreaking these independent souls can be a steep challenge for novice owners. American Foxhounds are good-natured, low-maintenance hounds who get on well with kids, dogs, even cats, but come with special considerations for prospective owners. They are closely associated with Revolutionary heroes and the rolling estates of old Virginia. """
   , """
   Nobs aren’t large dogs (a big male stands a shade below 20 inches) but their tough, sinewy bodies are built to withstand punishing terrain and harsh climates. They are a small-sized hunting dog of Spitz-type, which was thought to be extinct but survived as a farm and hunting dog in the Northern parts of Sweden and Finland. To be able to navigate the rough terrain and climate of the Scandinavian forests and hold large dangerous game, like moose, Norrbottenspets evolved to be extremely agile, rugged, and weatherproof with a fearless attitude, while at the same time kind and affectionate companions at home. They exhibit no extremes in physical characteristics because they must do all things well. The Norrbottenspets is a small, slightly rectangular spitzdog, well poised, with sinewy and well-developed muscles. Alert with head carried high, they have a fearless attitude and are extremely agile. They are calm, keen, and attentive, with a kind disposition. """
   ,""" 
   There are two Beagle varieties: those standing under 13 inches at the shoulder, and those between 13 and 15 inches. Both varieties are sturdy, solid, and “big for their inches,” as dog folks say. They come in such pleasing colors as lemon, red and white, and tricolor. The Beagle’s fortune is in his adorable face, with its big brown or hazel eyes set off by long, houndy ears set low on a broad head.A breed described as “merry” by its fanciers, Beagles are loving and lovable, happy, and companionable—all qualities that make them excellent family dogs. No wonder that for years the Beagle has been the most popular hound dog among American pet owners. These are curious, clever, and energetic hounds who require plenty of playtime. Not only is the Beagle an excellent hunting dog and loyal companion, it is also happy-go-lucky, funny, and—thanks to its pleading expression—cute. They were bred to hunt in packs, so they enjoy company and are generally easygoing."""  
]

temperament = [
  "Fearless, Alert, Fun-Loving",
  "Fearless, Fun-Loving, Proud",
  "Sweet, Patient, Devoted",
  "Affectionate, Loyal, Noble",
  "Confident, Clever, Lively",
  "Adaptable, Gentle, Smart",
  "Playful, Charming, Inquisitive",
  "Courageous, Good-Tempered, Canny",
  "Friendly, Independent, Amusing",
  "Playful but also Work-Oriented. Very Active and Upbeat."
]

def tokenize(text):
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Params: {text: String}
    Returns: List
    """
    # YOUR CODE HERE
    text = text.lower()
    result = re.findall("[a-z]+", text)
    return result


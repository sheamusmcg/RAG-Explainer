"""Pre-built document corpora for the RAG Explainer demo."""


def get_sample_corpora() -> dict:
    """Return a dict mapping corpus name to a list of document dicts.

    Each document: {"title": str, "text": str}
    """
    return {
        "Space Exploration": _SPACE_EXPLORATION,
        "Famous Scientists": _FAMOUS_SCIENTISTS,
    }


_SPACE_EXPLORATION = [
    {
        "title": "The Apollo Program",
        "text": (
            "The Apollo program was a NASA spaceflight program that ran from 1961 to 1972 "
            "and succeeded in landing the first humans on the Moon. Apollo 11, launched on "
            "July 16, 1969, carried astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. "
            "Armstrong and Aldrin spent about two and a quarter hours on the lunar surface while "
            "Collins orbited above. The program included a total of 17 missions, six of which "
            "successfully landed on the Moon. Apollo 13 famously survived an oxygen tank explosion "
            "and returned its crew safely to Earth. The program advanced rocketry, materials science, "
            "and computing. The Saturn V rocket, standing 363 feet tall, remains the most powerful "
            "rocket ever flown. Apollo's legacy includes thousands of technological spinoffs, from "
            "water purification systems to fireproof materials used by firefighters. The total cost "
            "of the program was approximately 25.4 billion dollars, equivalent to roughly 200 billion "
            "in today's currency. The program employed over 400,000 people at its peak and involved "
            "contributions from more than 20,000 companies and universities."
        ),
    },
    {
        "title": "Mars Exploration",
        "text": (
            "Mars has been a target of robotic exploration since the 1960s. NASA's Perseverance "
            "rover, which landed in Jezero Crater in February 2021, is searching for signs of "
            "ancient microbial life and collecting rock samples for future return to Earth. Its "
            "companion, the Ingenuity helicopter, became the first aircraft to achieve powered "
            "flight on another planet. Earlier rovers include Curiosity, which has been operating "
            "since 2012 and discovered evidence that Mars once had conditions suitable for life, "
            "including liquid water and organic molecules. The Mars Reconnaissance Orbiter has "
            "mapped the planet's surface in extraordinary detail, revealing ancient river valleys "
            "and mineral deposits that form only in water. Future plans include NASA's Mars Sample "
            "Return mission, which aims to bring Perseverance's collected samples back to Earth "
            "by the 2030s. SpaceX has outlined plans for crewed Mars missions using the Starship "
            "vehicle. The journey to Mars takes roughly seven months, and communication delays "
            "between Earth and Mars range from 4 to 24 minutes depending on orbital positions."
        ),
    },
    {
        "title": "The International Space Station",
        "text": (
            "The International Space Station is a modular space station in low Earth orbit, "
            "orbiting at roughly 250 miles above the surface. It is a joint project among five "
            "space agencies: NASA, Roscosmos, JAXA, ESA, and CSA. Assembly began in 1998 and the "
            "station has been continuously occupied since November 2000, making it the longest "
            "continuous human presence in space. The ISS travels at about 17,500 miles per hour "
            "and completes one orbit every 90 minutes, meaning the crew sees 16 sunrises and "
            "sunsets each day. The station serves as a microgravity research laboratory where "
            "scientists conduct experiments in biology, physics, astronomy, and materials science. "
            "Research on the ISS has led to advances in water purification, medical imaging, and "
            "understanding how the human body adapts to long-duration spaceflight. Astronauts "
            "typically serve six-month rotations and must exercise two hours daily to combat "
            "muscle and bone loss. The station measures 357 feet across and has a pressurized "
            "volume comparable to a five-bedroom house."
        ),
    },
    {
        "title": "The Hubble Space Telescope",
        "text": (
            "The Hubble Space Telescope was launched into orbit on April 24, 1990, aboard the "
            "Space Shuttle Discovery. Named after astronomer Edwin Hubble, it orbits Earth at "
            "about 340 miles altitude and has made over 1.5 million observations since its launch. "
            "Shortly after deployment, a flaw in its primary mirror caused blurred images. "
            "A dramatic repair mission in 1993 installed corrective optics, restoring the telescope "
            "to full capability. Hubble has made groundbreaking discoveries including determining "
            "the rate of expansion of the universe, providing evidence for dark energy, observing "
            "galaxies in their earliest stages, and capturing detailed images of distant nebulae "
            "and star-forming regions. The telescope can observe in ultraviolet, visible, and "
            "near-infrared wavelengths. Five servicing missions between 1993 and 2009 repaired "
            "and upgraded its instruments. Hubble's successor, the James Webb Space Telescope, "
            "launched in December 2021 and operates primarily in infrared wavelengths, allowing "
            "it to see even further back in time to the earliest galaxies."
        ),
    },
    {
        "title": "SpaceX and Commercial Spaceflight",
        "text": (
            "SpaceX, founded by Elon Musk in 2002, has transformed the space industry by "
            "developing reusable rocket technology. The Falcon 9 rocket's first stage can land "
            "itself after launch and be reflown, reducing launch costs dramatically. As of 2024, "
            "Falcon 9 boosters have been reused up to 20 times each. The Dragon spacecraft "
            "transports cargo and crew to the International Space Station, and in 2020 became "
            "the first commercial vehicle to carry astronauts to orbit. SpaceX's Starship, "
            "standing nearly 400 feet tall, is designed to be the most powerful rocket ever built "
            "and aims to enable crewed missions to Mars. The company also operates Starlink, a "
            "satellite internet constellation with over 5,000 satellites providing broadband "
            "coverage worldwide. Other commercial space companies include Blue Origin, founded "
            "by Jeff Bezos, which is developing the New Glenn orbital rocket and the Blue Moon "
            "lunar lander, and Rocket Lab, which provides dedicated small satellite launches "
            "with its Electron rocket. The commercial space sector has grown rapidly, with global "
            "space economy revenues exceeding 400 billion dollars annually."
        ),
    },
    {
        "title": "The Voyager Missions",
        "text": (
            "Voyager 1 and Voyager 2, launched in 1977, are the farthest human-made objects "
            "from Earth. Voyager 1, at over 15 billion miles away, entered interstellar space "
            "in 2012, becoming the first spacecraft to do so. Voyager 2 followed into interstellar "
            "space in 2018. Both spacecraft carry a Golden Record, a phonograph record containing "
            "sounds and images selected to portray the diversity of life on Earth, intended as a "
            "message to any extraterrestrial civilization that might find them. Voyager 2 remains "
            "the only spacecraft to have visited Uranus and Neptune, providing the first close-up "
            "images of these distant ice giants and their moons. The Voyager missions discovered "
            "23 new moons across the outer planets, active volcanoes on Jupiter's moon Io, and "
            "evidence of a subsurface ocean on Europa. Despite being over 45 years old, both "
            "spacecraft continue to send data back to Earth, though their plutonium power sources "
            "are gradually weakening. NASA estimates they will have enough power to communicate "
            "with at least one science instrument until around 2025. The radio signals from Voyager 1 "
            "take over 22 hours to reach Earth at the speed of light."
        ),
    },
]

_FAMOUS_SCIENTISTS = [
    {
        "title": "Isaac Newton",
        "text": (
            "Isaac Newton, born in 1643 in Woolsthorpe, England, is widely regarded as one of "
            "the most influential scientists in history. His work Philosophiae Naturalis Principia "
            "Mathematica, published in 1687, laid the foundations for classical mechanics. Newton "
            "formulated the three laws of motion and the law of universal gravitation, which "
            "explained how objects move on Earth and in space under the same set of rules. He "
            "developed calculus independently of Leibniz, invented the reflecting telescope, and "
            "conducted experiments on the nature of light, showing that white light is composed "
            "of a spectrum of colors. Newton served as Warden and Master of the Royal Mint, "
            "reforming England's currency, and was president of the Royal Society from 1703 until "
            "his death in 1727. His gravitational theory remained the dominant model of physics "
            "for over 200 years until Einstein's general relativity provided a more complete "
            "description. Newton famously said that if he had seen further, it was by standing "
            "on the shoulders of giants."
        ),
    },
    {
        "title": "Marie Curie",
        "text": (
            "Marie Curie, born Maria Sklodowska in Warsaw, Poland, in 1867, was a physicist "
            "and chemist who conducted pioneering research on radioactivity. She was the first "
            "woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two "
            "different sciences: Physics in 1903 (shared with Pierre Curie and Henri Becquerel) "
            "and Chemistry in 1911. She discovered the elements polonium and radium, developed "
            "mobile X-ray units during World War I that were used to diagnose injuries near the "
            "front lines, and established the Curie Institutes in Paris and Warsaw, which remain "
            "major research centers today. Her research on radioactivity, a term she coined, "
            "opened entirely new fields of study in physics, chemistry, and medicine. Curie faced "
            "significant discrimination as a woman in science but persisted through determination "
            "and brilliance. She died in 1934 from aplastic anemia, likely caused by long-term "
            "exposure to radiation. Her laboratory notebooks are still so radioactive that they "
            "must be stored in lead-lined boxes."
        ),
    },
    {
        "title": "Albert Einstein",
        "text": (
            "Albert Einstein, born in 1879 in Ulm, Germany, revolutionized physics with his "
            "theories of special and general relativity. His 1905 papers, sometimes called his "
            "annus mirabilis works, introduced special relativity, explained the photoelectric "
            "effect (for which he won the 1921 Nobel Prize in Physics), demonstrated the existence "
            "of atoms through Brownian motion analysis, and introduced the equation E equals mc "
            "squared, showing that mass and energy are interchangeable. General relativity, "
            "published in 1915, described gravity not as a force but as a curvature of spacetime "
            "caused by mass and energy. This theory predicted gravitational lensing, black holes, "
            "and gravitational waves, all of which have been confirmed by observation. Einstein "
            "fled Nazi Germany in 1933 and settled at the Institute for Advanced Study in Princeton, "
            "New Jersey. He became a vocal advocate for civil rights, pacifism, and nuclear "
            "disarmament. His work fundamentally changed our understanding of space, time, and "
            "the universe, and his name has become synonymous with genius."
        ),
    },
    {
        "title": "Alan Turing",
        "text": (
            "Alan Turing, born in 1912 in London, was a mathematician and computer scientist "
            "whose work laid the theoretical foundation for modern computing. His 1936 paper "
            "introduced the concept of the Turing machine, an abstract device that formalized "
            "the notion of computation and algorithm. During World War II, Turing worked at "
            "Bletchley Park where he played a crucial role in breaking the German Enigma cipher, "
            "which is estimated to have shortened the war by two or more years and saved millions "
            "of lives. He designed the Bombe, an electromechanical device used to find Enigma "
            "settings. After the war, Turing worked on early computer designs at the National "
            "Physical Laboratory and the University of Manchester. He proposed the Turing Test "
            "as a measure of machine intelligence, asking whether a machine could exhibit behavior "
            "indistinguishable from a human. He also made contributions to mathematical biology, "
            "studying how patterns like stripes and spots form in nature. Turing was prosecuted "
            "for his homosexuality in 1952 and died in 1954. He received a posthumous royal "
            "pardon in 2013."
        ),
    },
    {
        "title": "Stephen Hawking",
        "text": (
            "Stephen Hawking, born in 1942 in Oxford, England, was a theoretical physicist "
            "who made groundbreaking contributions to cosmology, general relativity, and quantum "
            "gravity. His most famous discovery, known as Hawking radiation, showed that black "
            "holes are not entirely black but emit thermal radiation due to quantum effects near "
            "the event horizon. This insight connected general relativity, quantum mechanics, and "
            "thermodynamics in a way that had never been done before. His book A Brief History "
            "of Time, published in 1988, sold over 25 million copies and made complex physics "
            "accessible to a general audience. Hawking was diagnosed with motor neuron disease "
            "at age 21 and was given two years to live, yet he continued working for over 50 more "
            "years. He held the Lucasian Chair of Mathematics at Cambridge University, a position "
            "once held by Newton. Hawking communicated using a speech-generating device after "
            "losing his ability to speak. He worked on the information paradox, the nature of "
            "the Big Bang, and unified theories of physics until his death in 2018 at age 76."
        ),
    },
]

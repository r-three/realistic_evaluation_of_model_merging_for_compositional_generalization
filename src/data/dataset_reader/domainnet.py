import logging
import os
from typing import Any, List


from src.data.dataset_reader.DatasetReader import DatasetReader
from src.data.DatasetConfig import DatasetConfig

furniture = {
    "bathtub": 0,
    "bed": 1,
    "bench": 2,
    "ceiling_fan": 3,
    "chair": 4,
    "chandelier": 5,
    "couch": 6,
    "door": 7,
    "dresser": 8,
    "fence": 9,
    "fireplace": 10,
    "floor_lamp": 11,
    "hot_tub": 12,
    "ladder": 13,
    "lantern": 14,
    "mailbox": 15,
    "picture_frame": 16,
    "pillow": 17,
    "postcard": 18,
    "see_saw": 19,
    "sink": 20,
    "sleeping_bag": 21,
    "stairs": 22,
    "stove": 23,
    "streetlight": 24,
    "suitcase": 25,
    "swing_set": 26,
    "table": 27,
    "teapot": 28,
    "toilet": 29,
    "toothbrush": 30,
    "toothpaste": 31,
    "umbrella": 32,
    "vase": 33,
    "wine_glass": 34,
}

mammal = {
    "bat": 0,
    "bear": 1,
    "camel": 2,
    "cat": 3,
    "cow": 4,
    "dog": 5,
    "dolphin": 6,
    "elephant": 7,
    "giraffe": 8,
    "hedgehog": 9,
    "horse": 10,
    "kangaroo": 11,
    "lion": 12,
    "monkey": 13,
    "mouse": 14,
    "panda": 15,
    "pig": 16,
    "rabbit": 17,
    "raccoon": 18,
    "rhinoceros": 19,
    "sheep": 20,
    "squirrel": 21,
    "tiger": 22,
    "whale": 23,
    "zebra": 24,
}

tool = {
    "anvil": 0,
    "axe": 1,
    "bandage": 2,
    "basket": 3,
    "boomerang": 4,
    "bottlecap": 5,
    "broom": 6,
    "bucket": 7,
    "compass": 8,
    "drill": 9,
    "dumbbell": 10,
    "hammer": 11,
    "key": 12,
    "nail": 13,
    "paint_can": 14,
    "passport": 15,
    "pliers": 16,
    "rake": 17,
    "rifle": 18,
    "saw": 19,
    "screwdriver": 20,
    "shovel": 21,
    "skateboard": 22,
    "stethoscope": 23,
    "stitches": 24,
    "sword": 25,
    "syringe": 26,
    "wheel": 27,
}

cloth = {
    "belt": 0,
    "bowtie": 1,
    "bracelet": 2,
    "camouflage": 3,
    "crown": 4,
    "diamond": 5,
    "eyeglasses": 6,
    "flip_flops": 7,
    "hat": 8,
    "helmet": 9,
    "jacket": 10,
    "lipstick": 11,
    "necklace": 12,
    "pants": 13,
    "purse": 14,
    "rollerskates": 15,
    "shoe": 16,
    "shorts": 17,
    "sock": 18,
    "sweater": 19,
    "t-shirt": 20,
    "underwear": 21,
    "wristwatch": 22,
}

eletricity = {
    "calculator": 0,
    "camera": 1,
    "cell_phone": 2,
    "computer": 3,
    "cooler": 4,
    "dishwasher": 5,
    "fan": 6,
    "flashlight": 7,
    "headphones": 8,
    "keyboard": 9,
    "laptop": 10,
    "light_bulb": 11,
    "megaphone": 12,
    "microphone": 13,
    "microwave": 14,
    "oven": 15,
    "power_outlet": 16,
    "radio": 17,
    "remote_control": 18,
    "spreadsheet": 19,
    "stereo": 20,
    "telephone": 21,
    "television": 22,
    "toaster": 23,
    "washing_machine": 24,
}

building = {
    "The_Eiffel_Tower": 0,
    "The_Great_Wall_of_China": 1,
    "barn": 2,
    "bridge": 3,
    "castle": 4,
    "church": 5,
    "diving_board": 6,
    "garden": 7,
    "garden_hose": 8,
    "golf_club": 9,
    "hospital": 10,
    "house": 11,
    "jail": 12,
    "lighthouse": 13,
    "pond": 14,
    "pool": 15,
    "skyscraper": 16,
    "square": 17,
    "tent": 18,
    "waterslide": 19,
    "windmill": 20,
}

office = {
    "alarm_clock": 0,
    "backpack": 1,
    "bandage": 2,
    "binoculars": 3,
    "book": 4,
    "calendar": 5,
    "candle": 6,
    "clock": 7,
    "coffee_cup": 8,
    "crayon": 9,
    "cup": 10,
    "envelope": 11,
    "eraser": 12,
    "map": 13,
    "marker": 14,
    "mug": 15,
    "nail": 16,
    "paintbrush": 17,
    "paper_clip": 18,
    "pencil": 19,
    "scissors": 20,
}

human_body = {
    "arm": 0,
    "beard": 1,
    "brain": 2,
    "ear": 3,
    "elbow": 4,
    "eye": 5,
    "face": 6,
    "finger": 7,
    "foot": 8,
    "goatee": 9,
    "hand": 10,
    "knee": 11,
    "leg": 12,
    "moustache": 13,
    "mouth": 14,
    "nose": 15,
    "skull": 16,
    "smiley_face": 17,
    "toe": 18,
    "tooth": 19,
}

road_transportation = {
    "ambulance": 0,
    "bicycle": 1,
    "bulldozer": 2,
    "bus": 3,
    "car": 4,
    "firetruck": 5,
    "motorbike": 6,
    "pickup_truck": 7,
    "police_car": 8,
    "roller_coaster": 9,
    "school_bus": 10,
    "tractor": 11,
    "train": 12,
    "truck": 13,
    "van": 14,
}

food = {
    "birthday_cake": 0,
    "bread": 1,
    "cake": 2,
    "cookie": 3,
    "donut": 4,
    "hamburger": 5,
    "hot_dog": 6,
    "ice_cream": 7,
    "lollipop": 8,
    "peanut": 9,
    "pizza": 10,
    "popsicle": 11,
    "sandwich": 12,
    "steak": 13,
}

nature = {
    "beach": 0,
    "cloud": 1,
    "hurricane": 2,
    "lightning": 3,
    "moon": 4,
    "mountain": 5,
    "ocean": 6,
    "rain": 7,
    "rainbow": 8,
    "river": 9,
    "snowflake": 10,
    "star": 11,
    "sun": 12,
    "tornado": 13,
}

cold_blooded = {
    "crab": 0,
    "crocodile": 1,
    "fish": 2,
    "frog": 3,
    "lobster": 4,
    "octopus": 5,
    "scorpion": 6,
    "sea_turtle": 7,
    "shark": 8,
    "snail": 9,
    "snake": 10,
    "spider": 11,
}

music = {
    "cello": 0,
    "clarinet": 1,
    "drums": 2,
    "guitar": 3,
    "harp": 4,
    "piano": 5,
    "saxophone": 6,
    "trombone": 7,
    "trumpet": 8,
    "violin": 9,
}

fruit = {
    "apple": 0,
    "banana": 1,
    "blackberry": 2,
    "blueberry": 3,
    "grapes": 4,
    "pear": 5,
    "pineapple": 6,
    "strawberry": 7,
    "watermelon": 8,
}

sport = {
    "baseball": 0,
    "baseball_bat": 1,
    "basketball": 2,
    "flying_saucer": 3,
    "hockey_puck": 4,
    "hockey_stick": 5,
    "snorkel": 6,
    "soccer_ball": 7,
    "tennis_racquet": 8,
    "yoga": 9,
}

tree = {
    "bush": 0,
    "cactus": 1,
    "flower": 2,
    "grass": 3,
    "house_plant": 4,
    "leaf": 5,
    "palm_tree": 6,
    "tree": 7,
}

bird = {
    "bird": 0,
    "duck": 1,
    "flamingo": 2,
    "owl": 3,
    "parrot": 4,
    "penguin": 5,
    "swan": 6,
}

vegetable = {
    "asparagus": 0,
    "broccoli": 1,
    "carrot": 2,
    "mushroom": 3,
    "onion": 4,
    "peas": 5,
    "potato": 6,
    "string_bean": 7,
}

shape = {
    "circle": 0,
    "hexagon": 1,
    "line": 2,
    "octagon": 3,
    "squiggle": 4,
    "triangle": 5,
    "zigzag": 6,
}

kitchen = {
    "fork": 0,
    "frying_pan": 1,
    "hourglass": 2,
    "knife": 3,
    "lighter": 4,
    "matches": 5,
    "spoon": 6,
    "wine_bottle": 7,
}

water_transportation = {
    "aircraft_carrier": 0,
    "canoe": 1,
    "cruise_ship": 2,
    "sailboat": 3,
    "speedboat": 4,
    "submarine": 5,
}

sky_transportation = {
    "airplane": 0,
    "helicopter": 1,
    "hot_air_balloon": 2,
    "parachute": 3,
}

insect = {"ant": 0, "bee": 1, "butterfly": 2, "mosquito": 3}

others = {
    "The_Mona_Lisa": 0,
    "angel": 1,
    "animal_migration": 2,
    "campfire": 3,
    "cannon": 4,
    "dragon": 5,
    "feather": 6,
    "fire_hydrant": 7,
    "mermaid": 8,
    "snowman": 9,
    "stop_sign": 10,
    "teddy-bear": 11,
    "traffic_light": 12,
}
CATEGORIES = {
    "furniture": furniture,
    "mammal": mammal,
    "tool": tool,
    "cloth": cloth,
    "electricity": eletricity,
    "building": building,
    "office": office,
    "human_body": human_body,
    "road_transportation": road_transportation,
    "food": food,
    "nature": nature,
    "cold_blooded": cold_blooded,
    "music": music,
    "fruit": fruit,
    "sport": sport,
    "tree": tree,
    "bird": bird,
    "vegatable": vegetable,
    "shape": shape,
    "kitchen": kitchen,
    "water_transportation": water_transportation,
    "sky_transportation": sky_transportation,
    "insect": insect,
    "others": others,
}

CATEGORY_GLOBAL_IDX = {
    "furniture": 0,
    "mammal": 35,
    "tool": 60,
    "cloth": 88,
    "electricity": 111,
    "building": 136,
    "office": 157,
    "human_body": 178,
    "road_transportation": 198,
    "food": 213,
    "nature": 227,
    "cold_blooded": 241,
    "music": 253,
    "fruit": 263,
    "sport": 272,
    "tree": 282,
    "bird": 290,
    "vegatable": 297,
    "shape": 305,
    "kitchen": 312,
    "water_transportation": 320,
    "sky_transportation": 326,
    "insect": 330,
    "others": 334,
}

domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


class DomainNetReader(DatasetReader):
    """
    DatasetReader that constructs dataset from reading a local text file
    """

    def __init__(self, dataset_config: DatasetConfig):
        """
        Args:
            dataset_config:
        """
        self.dataset_config = dataset_config

        # Cached the dataset and temaplates
        self.cached_originalData = {}
        self.cached_datasets = {}

    def _get_originalData(self) -> List[Any]:
        """
        Get the data in its original format

        Returns:
            load_data:
        """
        assert self.dataset_config.template_idx is None, (
            self.dataset_config.dataset_name
            + "images has not but template "
            + self.dataset_config.template_idx
            + " was specified"
        )

        assert self.dataset_config.instruction_format == "no_instructions", (
            self.dataset_config.instruction_format + "should have no instructions"
        )

        load_data = []

        input_filepath = (
            f"{self.dataset_config.domain}_{self.dataset_config.split}_fold.txt"
        )
        dataset_dir = "datasets/DomainNet"
        with open(os.path.join(dataset_dir, input_filepath), "r") as f:
            for line in f.readlines():
                tab_split = line.strip().split()
                image_path = os.path.join(dataset_dir, tab_split[0])

                class_name = image_path.split("/")[3]

                if class_name in CATEGORIES[self.dataset_config.task]:
                    lbl = int(CATEGORIES[self.dataset_config.task][class_name])

                    if self.dataset_config.shift_lbls:
                        true_lbl = CATEGORY_GLOBAL_IDX[self.dataset_config.task] + lbl
                    else:
                        true_lbl = lbl

                    load_data.append(
                        {
                            "image_path": image_path,
                            "lbl": true_lbl,
                        }
                    )

        return load_data

    def get_dataset(self, train_or_eval) -> List[Any]:
        """
        Returns:
            dataset
        """
        if self.dataset_config not in self.cached_datasets:
            original_data = self._get_originalData()
            # Trim the original data smaller
            if self.dataset_config.max_number_of_samples != -1:
                assert (
                    self.dataset_config.template_idx != -2
                ), f"Cannot handle max number of samples if doing a cross product of samples and templates "
                original_data = original_data[
                    : self.dataset_config.max_number_of_samples
                ]
            logging.info(
                f"Loaded {self.dataset_config.split} which contains {len(original_data)} datapoints"
            )
            self.cached_datasets[self.dataset_config] = original_data
        return self.cached_datasets[self.dataset_config]

    def get_datasetMetrics(self) -> List[str]:
        return ["Accuracy"]


def getPairs_taskWithDomain():
    taskDomain_pairs = []

    for task_idx, (task, _) in enumerate(CATEGORIES.items()):
        domain_idx = task_idx // 4
        domain = domains[domain_idx]
        taskDomain_pairs.append((task_idx, task, domain))

    return taskDomain_pairs


def getCrossProductPairs_taskWithDomain():
    taskDomain_pairs = []

    for task_idx, (task, _) in enumerate(CATEGORIES.items()):
        for domain_idx, domain in enumerate(domains):
            domain = domains[domain_idx]
            taskDomain_pairs.append((task_idx, task, domain))

    return taskDomain_pairs


def getMissingPairs_taskWithDomain():
    taskDomain_pairs = []

    for task_idx, (task, _) in enumerate(CATEGORIES.items()):
        paired_domainIdx = task_idx // 4
        for domain_idx, domain in enumerate(domains):
            if domain_idx != paired_domainIdx:
                domain = domains[domain_idx]
                taskDomain_pairs.append((task_idx, task, domain))

    return taskDomain_pairs


if __name__ == "__main__":
    start_idx = 0
    all_classes = set()
    for category, classes in CATEGORIES.items():
        print(category, start_idx)
        start_idx += len(classes)
        all_classes.update(set(classes.keys()))

    with open("datasets/DomainNet/all_train_fold.txt", "r") as f:
        for line in f.readlines():
            class_lbl = line.split("/")[1]

            preprocess_class_label = class_lbl.replace("_", " ")

            if preprocess_class_label not in all_classes:
                import ipdb

                ipdb.set_trace()

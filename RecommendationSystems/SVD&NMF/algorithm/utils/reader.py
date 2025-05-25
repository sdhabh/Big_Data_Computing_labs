
from .builtin_datasets import BUILTIN_DATASETS


class Reader:

    def __init__(
        self,
        name=None,
        line_format="user item rating",
        sep=None,
        rating_scale=(1, 5),
        skip_lines=0,
    ):

        if name:
            try:
                self.__init__(**BUILTIN_DATASETS[name].reader_params)
            except KeyError:
                raise ValueError(
                    "unknown reader "
                    + name
                    + ". Accepted values are "
                    + ", ".join(BUILTIN_DATASETS.keys())
                    + "."
                )
        else:
            self.sep = sep
            self.skip_lines = skip_lines
            self.rating_scale = rating_scale

            lower_bound, higher_bound = rating_scale

            splitted_format = line_format.split()

            entities = ["user", "item", "rating"]
            if "timestamp" in splitted_format:
                self.with_timestamp = True
                entities.append("timestamp")
            else:
                self.with_timestamp = False

            # check that all fields are correct
            if any(field not in entities for field in splitted_format):
                raise ValueError("line_format parameter is incorrect.")

            self.indexes = [splitted_format.index(entity) for entity in entities]

    def parse_line(self, line):

        line = line.split(self.sep)
        try:
            if self.with_timestamp:
                uid, iid, r, timestamp = (line[i].strip() for i in self.indexes)
            else:
                uid, iid, r = (line[i].strip() for i in self.indexes)
                timestamp = None

        except IndexError:
            raise ValueError(
                "Impossible to parse line. Check the line_format" " and sep parameters."
            )

        return uid, iid, float(r), timestamp

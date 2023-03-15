"""Import modules"""
import os
import typing
import warnings
import itertools
import geopandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


class Agros:
    """
    Class designed to illustrate data from the Our World in Data
    agricultural dataset, based on countries. Country names from agricultural
    data should be used.

    @Class attributes
    data: pandas dataframe
        Pandas DataFrame containing all data from the csv file
    geodata: geopandas dataframe
        Geopandas dataframe from shapefile
    all_data: geopandas dataframe
        geopandas dataframe containing all information from dataset

    @Private class attributes
    _url: str
        String link to csv data
    _geourl: str
        String to naturalearth dataset
    _merge_dict: dictionary
        mapping of countries to rename in Geodata.csv
    _exclude: list:str
        Countries to exclude from aggregation

    @Public methods
    download_data(None) -> None
        Gets data from agricultural and vector database
    countries_list(None) -> list
        Returns all unique countries in dataset
    variable_correlation(None) -> None
        Plots correlation between all countries
    area_chart(country: str, normalize: bool) -> None
        Plots data from all ouput columns for selected country
    total_output_country(*country: str) -> None
        Plots total agricultural ouput from selected countries
    gapminder(year: int) -> None
        Plots the output quantity based on the fertilizer quantity
    cloropleth(year: int) -> None
        Plots tfp column on chloropleth on a given year
    arima_grid_search(self, data: list) -> typing.Tuple
        Gives best ARIMA parameters to predictor
    predictor( *countries: str) -> None
        Generate a plot of predicted TFP values for the specified countries.
    """

    def __init__(self) -> None:
        """
        Initialize Agros class with data from sources

        @params
        None

        @Returns
        None

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        Natural Earth: 1:110m cultural data
        """

        self.data = pd.DataFrame()
        self.geodata = geopandas.GeoDataFrame()
        self.all_data = geopandas.GeoDataFrame()

        self._geourl = "naturalearth_lowres"

        self._url = (
            "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/"
            "Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total"
            "%20factor%20productivity%20(USDA).csv?=raw=true"
        )

        self._merge_dict = {
            "United States of America": "United States",
            "Dem. Rep. Congo": "Democratic Republic of Congo",
            "Central African Rep.": "Central African Republic",
            "S. Sudan": "South Sudan",
        }

        self._exclude = [
            "World",
            "Western Europe",
            "West Asia",
            "West Africa",
            "Upper-middle income",
            "Sub-Saharan Africa",
            "Southeast Asia",
            "Southern Africa",
            "Southern Europe",
            "South Africa",
            "South Asia",
            "Northeast Asia",
            "Northern Europe",
            "North Africa",
            "North America",
            "Low income",
            "Lower-middle income",
            "Least developed countries",
            "Latin America and the Caribbean",
            "Horn of Africa",
            "High Income",
            "Former Soviet Union",
            "Europe",
            "East Africa",
            "Eastern Europe",
            "Developed Asia",
            "Developed countries",
            "Caribbean",
            "Central Africa",
            "Oceania",
            "Micronesia",
            "Central America",
            "Central Asia",
            "Central Europe",
            "Asia",
            "High income",
            "Pacific",
        ]

    def download_data(self) -> None:
        """
        Method gets data to class, either reading it from file, or downloading file.
        If both fail the method returns empty dataframe.

        @params
        None

        @Returns
        None

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        Natural Earth: 1:110m cultural data
        """

        # Use './' for showcase, '../' for testing .py
        file_path_base = "./downloads/"
        file_path_agriculture = "./downloads/Data.csv"
        file_path_geographical = "./downloads/Geodata.shp"

        if not os.path.exists(file_path_base):
            print(f"Setting up folder {file_path_base}")
            os.mkdir(file_path_base)

        if os.path.exists(file_path_agriculture):
            print(f"Reading agricultural data from {file_path_agriculture}")
            self.data = pd.read_csv(file_path_agriculture)
        else:
            self.data = pd.read_csv(self._url)
            print(
                f"Downloading and saving agricultural data to {file_path_agriculture}"
            )
            self.data.to_csv(file_path_agriculture, encoding="utf-8")

        if os.path.exists(file_path_geographical):
            print(f"Reading geodata data from {file_path_geographical}")
            self.geodata = geopandas.read_file(file_path_geographical)
        else:
            self.geodata = geopandas.read_file(
                geopandas.datasets.get_path(self._geourl)
            )
            print(f"Downloading and saving geodata to {file_path_geographical}")
            self.geodata.to_file(file_path_geographical)

        geodata_renamed = self.geodata.copy()
        geodata_renamed["name"].replace(self._merge_dict, inplace=True)
        self.all_data = geodata_renamed.rename(columns={"name": "Entity"})\
                                        .merge(self.data, on="Entity")

        print("All files read successfully")

    def total_output_country(self, *country: str) -> None:
        """
        This method compars the total of the "_output_" columns for countries and plot it.
        Only input one of the parameters, single string will be transformed to list.

        @params
        country: String with country name or countries separated by commas

        @Returns
        None

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        """

        if not country:
            raise KeyError("Please input one country or more")

        all_countries = self.countries_list()

        for value in country:
            if value not in all_countries:
                raise TypeError(f"Country {value} not found in data")

        target_columns = [
            "Entity",
            "Year",
            "crop_output_quantity",
            "animal_output_quantity",
            "fish_output_quantity",
        ]

        relevant_data = self.data[self.data["Entity"].isin(country)][target_columns]

        relevant_data["total_output"] = relevant_data[
            ["crop_output_quantity", "animal_output_quantity", "fish_output_quantity"]
        ].sum(axis=1)

        result = (
            relevant_data[["Year", "Entity", "total_output"]]
            .pivot_table(index="Year", columns="Entity", values="total_output")
            .reset_index()
        )

        title_header = ", ".join(country)
        result.plot.line(x="Year", title=f"Total output per year {title_header}")
        plt.annotate(
            text="Source: Our World in Data, Feb 1, 2022",
            xy=(1, 1),
            xycoords="figure points",
        )
        plt.tight_layout()
        plt.show()

    def countries_list(self) -> list:
        """
        Method that outputs a list of countries available in the data.

        @params
        None

        @Returns
        countries: List of unique countries in the data

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        """

        countries = list(dict.fromkeys(self.data["Entity"]))
        countries = [i for i in countries if i not in self._exclude]
        return countries

    def variable_correlation(self) -> None:
        """
        This function takes the data and plots a heatmap,
        that shows the correlation between the "_quantity" columns.

        @params
        None

        @Returns
        None

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        """
        
        quantity_columns = self.data.filter(regex="_quantity$")

        corr_matrix = quantity_columns.corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        plt.figure(dpi=100)
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True)
        plt.annotate(
            text="Source: Our World in Data, Feb 1, 2022",
            xy=(1, 1),
            xycoords="figure points",
        )
        plt.tight_layout()
        plt.show()

    def area_chart(self, country: str = None, normalize: bool = True) -> None:
        """
        Creates an area plot for the selected country

        @params
        country: string
            The country which data is to be plotted.
            If no country is stated, or "The world" is entered,
            the data will be aggregated across countries

        normalize: Boolean
            Whether to normalize the data

        @Returns
        A stackplot

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        """

        output_columns = [
            "Year",
            "Entity",
            "crop_output_quantity",
            "animal_output_quantity",
            "fish_output_quantity",
        ]
        relevant_df = self.data[output_columns]

        # Do aggreagation
        if country in [None, "World"]:
            grouped_df = relevant_df.groupby(["Year"], as_index=False).sum()

            # Check for and calculate normalization if wanted
            if normalize:
                total = np.array(
                    [
                        grouped_df["crop_output_quantity"],
                        grouped_df["animal_output_quantity"],
                        grouped_df["fish_output_quantity"],
                    ]
                )
                percent_y = total / total.sum(axis=0).astype(float) * 100

                fig = plt.figure()
                axes = fig.add_subplot(111)
                axes.stackplot(grouped_df["Year"], percent_y)
                axes.set_title("Yearly percentage output for the world")
                axes.set_ylabel("Percent (%)")
                axes.margins(0, 0)
                plt.legend(output_columns[2:])
                plt.annotate(
                    text="Source: Our World in Data, Feb 1, 2022",
                    xy=(1, 1),
                    xycoords="figure points",
                )
                plt.tight_layout()
                plt.show()

            # Normal plot
            else:
                grouped_df.plot.area(x="Year")
                plt.legend(output_columns[2:])
                plt.title("Outputs aggregated for all countries")
                plt.tight_layout()
                plt.show()

        else:
            if country not in relevant_df["Entity"].value_counts().index:
                raise KeyError("The country does not exist")

            # Subset country data
            country_df = relevant_df[relevant_df["Entity"] == country]

            # Check for and calculate normalization if wanted
            if normalize:
                total = np.array(
                    [
                        country_df["crop_output_quantity"],
                        country_df["animal_output_quantity"],
                        country_df["fish_output_quantity"],
                    ]
                )
                percent_y = total / total.sum(axis=0).astype(float) * 100

                fig = plt.figure()
                axes = fig.add_subplot(111)
                axes.stackplot(country_df["Year"], percent_y)
                axes.set_title(f"Yearly percentage output for {country}")
                axes.set_ylabel("Percent (%)")
                axes.margins(0, 0)
                plt.legend(output_columns[2:])
                plt.annotate(
                    text="Source: Our World in Data, Feb 1, 2022",
                    xy=(1, 1),
                    xycoords="figure points",
                )
                plt.tight_layout()
                plt.show()

            # Normal plot
            else:
                country_df.plot.area(x="Year")
                plt.title(f"Outputs for {country}")
                plt.tight_layout()
                plt.show()

    def gapminder(self, year: int = None) -> None:
        """
        Method that outputs a scatterplot visualization
        of the output quantity based on the fertilizer quantity.

        @params
        Year: int
            Year to plot

        @Returns
        Scatterplot of World's Agricultural Production in Year

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        """

        if not isinstance(year, int):
            raise TypeError("Year must be an integer")

        data = self.data[self.data["Year"] == year]

        plt.figure(dpi=125)
        sns.scatterplot(
            x=data["fertilizer_quantity"],
            y=data["output_quantity"],
            size=data["ag_land_quantity"],
            hue=data["Entity"],
            legend=False,
        )
        plt.grid(True)
        # Putting the x and y axis on a log scale for better visualization
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Fertilizer Quantity (log)", fontsize=12)
        plt.ylabel("Output Quantity (log)", fontsize=12)
        plt.title(f"World's Agricultural Production in {year}", fontsize=16)
        plt.annotate(
            text="Source: Our World in Data, Feb 1, 2022",
            xy=(1, 1),
            xycoords="figure points",
        )
        plt.tight_layout()
        plt.show()

    def chloropleth(self, year: int = None) -> None:
        """
        Plots cloropleth map

        @Param
        year: int

        @Returns
        none

        @Source
        Our World in Data: Agricultural total factor productivity (USDA)
        Natural Earth: 1:110m cultural data
        """

        # Raise (not really nesseccary but required)
        if not isinstance(year, int):
            raise TypeError("Value given not int")

        self.all_data[self.all_data["Year"] == year].plot(
            column="tfp",
            legend=True,
            figsize=[20, 10],
            legend_kwds={"label": "tfp value"},
        )

        plt.title(f"World map {year}")
        plt.annotate(
            text="Source: Our World in Data, Feb 1, 2022 & Natural Earth, March 2, 2023",
            xy=(1, 1),
            xycoords="figure points",
        )
        plt.tight_layout()
        plt.show()

    def arima_grid_search(self, data: list) -> typing.Tuple:
        """
        Grid search to find the best ARIMA model parameters for the given data.

        @Args
        data: pandas.Series
            Time series data to be fitted with ARIMA models.

        @Returns
        best_params: tuple
            A tuple containing the best ARIMA model parameters.
        """
        p_values = [0, 1, 2]
        d_values = range(0, 2)
        q_values = range(0, 2)

        grid = itertools.product(p_values, d_values, q_values)

        best_aic = float("inf")
        best_params = None

        for params in grid:
            model = ARIMA(data, order=params)
            results = model.fit()

            if results.aic < best_aic:
                best_aic = results.aic
                best_params = params

        return best_params

    def tfp_predictor(self, *countries: str) -> None:
        """
        Generate a plot of predicted TFP values for the specified countries
        using ARIMA modelling.

        @Args
        *countries: List
            of country names to include in the plot. Max 3

        @Returns
        None
        """

        available_countries = self.countries_list()

        valid_countries = [
            country for country in countries if country in available_countries
        ]

        if not valid_countries:
            raise ValueError(
                "None of the specified countries are available in the data."
                + f"\nAvailable countries are:\n{', '.join(available_countries)}"
            )

        if len(valid_countries) > 3:
            print(
                f"Warning: Only the first 3 valid countries ({', '.join(valid_countries[:3])})"
                + "will be considered. The rest will be ignored."
            )

        # Define the colors
        num_colors = min(len(valid_countries), 3)
        color_cycle = ["blue", "orange", "green"][:num_colors]
        country_colors = dict(zip(valid_countries, color_cycle))
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_cycle)

        for country in valid_countries[:3]:
            country_data = self.data[self.data["Entity"] == country.title()]
            tfp_data = country_data[["Year", "tfp"]].dropna()
            tfp_data["Year"] = pd.to_datetime(tfp_data["Year"], format="%Y")
            tfp_data.set_index("Year", inplace=True)
            tfp_data.index.freq = "YS"
            best_params = self.arima_grid_search(tfp_data)
            plt.plot(
                tfp_data.loc[:"2019"].index,
                tfp_data.loc[:"2019"]["tfp"],
                linestyle="-",
                label=country.title(),
                color=country_colors[country],
            )
            future_years = pd.date_range(start="2020", end="2050", freq="Y")
            model = ARIMA(tfp_data, order=best_params)
            model_fit = model.fit()
            predictions = model_fit.predict(start="2020", end="2050")
            predictions = predictions[:-1]
            plt.plot(
                future_years, predictions, linestyle=":", color=country_colors[country],
            )

        plt.xlabel("Year")
        plt.ylabel("TFP")
        plt.title("TFP Prediction Country Comparison")
        plt.legend()
        plt.annotate(
            text="Source: Our World in Data, Feb 1, 2022 & Natural Earth, March 2, 2023",
            xy=(1, 1),
            xycoords="figure points",
        )
        plt.tight_layout()
        plt.show()

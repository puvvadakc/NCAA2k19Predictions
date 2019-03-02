# NCAA2k19Predictions
UW INFO 370: Introduction to Data Science 
Final Project 2019

**Project Description**

**Purpose**
**       ** The overarching purpose of this project is to apply machine learning algorithms to a complex field which would benefit greatly from more accurate predictive models. Sports generally have long been an area ripe for data analytics, Baseball&#39;s sabermetrics revolution being perhaps the best example. But as yet, NCAA basketball remains an unpredictable, seemingly luck based sport where the winner is some combination of consistent, lucky and skilled. Our goal with this project is to attempt to reduce a basketball game to its components (like scoring, rebounding, blocks and steals) with the purpose of being able to accurately predict the outcome of a game given a series of inputs.
**       ** This research is of particular importance for the sport of basketball as an accurate predictive model would allow teams to not only predict the outcomes of their games, but also to analyze what areas they could improve in to better their results. This would likely have an impact not only on the college (NCAA) level, but also at other levels of the game.
        Additionally, this project is important as it explores whether or not sporting events can be predicted with any level of accuracy. Sports have long been elusive to predict, hence the strong gambling market surrounding them. The combination of skill and luck makes sports by nature difficult to predict. Nowhere is this better evidence than the NCAA tournament. Each year, 68 teams compete for a national title. The teams, selected and seeded by a special committee, are drawn from the 351 Division 1 college basketball teams. Nominally, the committee selects teams based on factors including but not limited to: wins, strength of schedule, conference record and, &quot;various computer metrics&quot;. All of these factors are based on regular season play. Once in the tournament, teams play against each other until a national champion is crowned (67 games total). Assuming that the odds of picking a game correctly are .5 (½), the odds of getting every single game correct in the tournament is 6.7762636e-21. Obviously, those odds are not great. As such, our mission is to attempt to create a series of algorithms that allow us to use both regular season statistics, and information about seeding, to predict NCAA Tournament games with better accuracy.

**Research
       ** Since sports predicting is very popular, we conducted some research to understand the current state of the field. We are not the first to use data science and machine learning to sports predict, yet the March Madness bracket is yet to be predicted accurately, so starting from scratch is a difficult option. We intended to find inspiration and ideas we can build on and tweak to make a more effective prediction model through our research.Three of the most applicable research papers we explored are explained below:

[Sports Predictions Machine Learning Framework] [https://www.sciencedirect.com/science/article/pii/S2210832717301485](https://www.sciencedirect.com/science/article/pii/S2210832717301485)

        The main purpose of this research paper is exploring sports predictions with machine learning, specifically through Artificial Neural Networks. The article mainly proposes and encouraging the following framework when constructing a proper predictive model.

 ![](https://ars.els-cdn.com/content/image/1-s2.0-S2210832717301485-gr3.jpg)

With sports predicting being a very complicated and lucrative space, many are moving towards utilizing machine learning for their predictive models, migrating from the previously popular statistical models. The main intention is to make the iteration  process of models more efficient, by re-analyzing and determining covariates, and making the proper adjustments to improve the model in the future iteration, with the idea being that a machine could be able to understand the nuances of the covariates better to always propose a more efficient model through each iteration.

[Comparison on Sports Forecast Accuracy]
[https://onlinelibrary.wiley.com/doi/epdf/10.1002/for.1091](https://onlinelibrary.wiley.com/doi/epdf/10.1002/for.1091)

The following article mainly focused on comparing the forecast accuracies of multiple methods which were prediction markets, tipsters, betting odds, and weight and rule based combinations of the previous forecasts. The results show that prediction markets and betting odds showed to have the best rates of success when forecasting games. We mainly used the information provided in this paper to explore prediction markets and betting odds. This is so that we can find useful aspects that we can incorporate into our own model to optimize the success rate of predictions.

[NCAA Sports Forecasting]
[https://core.ac.uk/download/pdf/7008571.pdf](https://core.ac.uk/download/pdf/7008571.pdf)

This research paper explores predicting NCAA games on a round by round basis, and they elaborated on an already existing prediction model called the Boulier-Stekler method, which they adjusted to predict for each round. The prediction model we intend to create is based even more specifically on. With their adjusted model, the team was able to accurately predict the overall winner and the first and second round of the bracket accurately, but their ability to predict round 4 especially was basically chance. This brings up the idea that different rounds may have different dynamics in accurately predicting. Also the overall success rate in predictions that the research group was able to receive was 73.8% which is better than blankly guessing each game outcome, but still would no where near suffice the accuracy needed to accurately predict every game of the tournament.

**Hypotheses**
The following are our Null and Alternative Hypotheses for this exploration in predicting NCAA March Madness:

H0: There is no relationship between a team&#39;s regular season stats or seeding and the result of their games in the NCAA Tournament

HA: There is a relationship between a team&#39;s regular season statistics or seeding and the result of their games in the NCAA Tournament.

**Datasets
       ** The following sources is where we will be grabbing the data that is necessary for our exploration:

[Season Results] ([http://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&amp;rpt=archive&amp;sportCode=MBB](http://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&amp;rpt=archive&amp;sportCode=MBB))

NCAA.org has downloadable csvs for every year since 2001, they list Rank, school name, number of Game, number of wins and losses, total points scored, and points scored against, and average points per game, scoring margins, three point shots made, two point shots made, three point attempts, two point attempts, total steals, number of fouls. These are all games for the year including the post season games. These are between half the games in the tournament. Half of these games should be in conference games and half are with whoever the individual teams can schedule games with. Looking at any statistics like wins, or points has a different meaning depending on their opponents which has a lot to do with their conference.

[Tournament Results]
([https://data.world/michaelaroy/ncaa-tournament-results](https://data.world/michaelaroy/ncaa-tournament-results))

This csv has the year, round, seed of the teams, region, and the scores, for all of the games in NCAA tournaments since 1985. The winners of each of the 32 conferences are invited to play in the final, and 36 other teams are invited by a selection committee. The selection committee selects them based on their regular season results with other teams. This ranking determines who they play during the tournament. In general the lowest ranked teams are the ones who won the lower ranked conferences. The tournament is single elimination.

Player data: for each season we can get a csv that contains a list of all the players for all of the teams. Because this is college basketball players are limited to player for 4 years then they graduate. This makes team&#39;s performance fluctuate as they lose and pick up players of varying skill. This can also be found on the ncaa website with the season results.

**Statistical and Machine Learning Methods
       ** We will use a neural network to test our hypothesis that certain features of the NCAA teams dataset would be positively correlated to wins and bracket standings. Sport prediction is usually treated as a classification problem, with one class, win, lose, or draw, to be predicted. The aim of classification is to predict a target variable, or class, by building a classification model based on a training dataset, and then utilizing that model to predict the value of the class of test data.  An artificial neural network usually contains interconnected components called neurons that transform a set of inputs into a desired output, mimicking the neural structure of the human brain. A neural network can dynamically adjust the weights given to certain features when building the classification model and is thus able to accomplish high levels of predictive accuracy. While this constant changing of weights might have complications due to overfitting or computational complexity, neural networks are still very flexible and would fit a model such as the NCAA due to the large amounts of different predictive variables.

**Target Audience
       ** Our target audience would be sports betting and gambling companies as well as consumers that want to try and earn money and win bracket competitions. Our resource would give detailed predictions as to how certain teams in the NCAA bracket would do and would allow our audience to make much more informed decisions on whether or not they want to bet money on certain games our outcomes, increasing their economic potential.
**       ** Our audience will learn what makes a given NCAA team more or less likely to win a certain number of games and how that changes with different variables like seeding, location, and opponent. Many fans might want to know what combination of factors is most likely to provide an accurate prediction of the NCAA bracket and how they can do better in deciding who to bet on.

**Technical Description**

**Web Resource Format
       ** The final web resource will be a HTML page hosted through github pages. Since we will be using Python 3 and Python packages to conduct the exploration regressions and analysis, we will be creating a Jupyter Python Notebook report that generates an HTML view when executed. Our report will be written with Markdown text formatting and with Latex for representing an mathematical functions and equations.

**Data Challenges
       ** A major data management challenge will be matching all of the team names up is multiple data sets. Pre season and post season may list the teams different ways, like by university or by team name or by city. We may need to go in and fix all of the team names manually.
        The csvs that NCAA.com aren&#39;t actually just csvs. Each .csv includes multiple csvs separated with with titles, they need to be broken up into different csv files before being used in analysis. The data in the different files will also have some repetition so it will need to be
        This collection of tournament results has a chance of containing small typos in the score or perhaps team name switches. A check that we can run to make sure that they have a high likelihood of being right is to test if the team that won each game moves on, while the team that lost doesn&#39;t appear again, because in single elimination they are gone.
        Dealing with the player data will be challenging because there are so many players, for each of the teams there are 10 players, who each have a different level of contribution to the way that the game plays. NCAA indexes the players individually so to link them up to the teams we will have to collect all of the members of the team and extract information from them to add to the team data prediction matrix.

**Necessary Technical Skills
       ** To effectively complete this project we will need to further develop our machine learning and data wrangling skills. One of the most important skills that we will develop is our ability to access datasets from an online source in an efficient manner. Additionally, we will be developing our understanding of the different machine learning approaches (K nearest neighbors and tree classifiers specifically) with the purpose of using these approaches to achieve our desired result.

**Analysis and Modelling Approach
       ** Our analysis will consist of constructing a model that uses regular season results of each of the teams, including their points scored and conference that they are in, to predict each teams outcome during that years march madness tournament. We have access to the last 17 years of results to build at test the model with. We are building a neural net model which uses supervised machine learning to weight different predictors in a way that a combination of them will lead to the most accurate outcome over a number of trials. We will split our data into testing years and training years. Using our more recent years as testing years we can hopefully get a model that works well as a future predictor. For our final predictions well will use all of the years of data to build a model then use the most recent seasons data to predict 2019s results. To form the best predictions we will need to undergo transformations to our data. A couple data transformations that would be useful are:
        Ranking the conferences with each other, will be important because a team doing well in a bad conference will likely lose to a team that is doing mediocre in a great conference, if we rank the conferences the points and win-loss records will be more usable in the model.
        We could also look at the number of player and staff changes for each of the teams, if we have this we might also be able to use the performance last year as a prediction, if a team did well the year before and has all the same people on it it will likely do well the next year too. Making this data will require us to check year vs year on the players data set. We will also want to combine this with total minutes played to see how much of an impact he had on the team on the court.
        With these and more predictors that we discover later we will put all of the data into a single dataframe with teams as rows, and all of the predictors and columns. After normalizing the data which is very important for use with a neural net, we will split off some of the years to use as testing years, then use the sklearn MLPClassifier which uses multilayer perceptron to give each input feature a weight. We will configure the mlpclassifier and run more prediction tests where we check how close the model can predict the results on the test years. When we find the configuration of the mlp that gives us the most accurate predictions we will recreate it using all of the data then use it to predict the scores for the 2019 championship.

**Major Challenges
        **One of the main things we will focus on is making sure that our algorithms are not overfitting. This would be particularly bad as the goal of the project is the find an algorithmic solution to predict** any** NCAA basketball game. If a model were to be overfitted it would lose much of its value.
        We will need to research and keep track of the differences in game play within the preseason, season, and postseason, since coaching changes, and the variables to keep track of for efficiency of game winning, might change for different parts of the basketball season. We will have to do detailed research on each covariate in our data to see how the they each trend with game winning. We may also need to conduct separate statistical analyses to determine optimal ranges. With games having so many different aspects that can dictate the outcome, we will have to consider many covariates within our model for prediction, like fouling, scoring, defence, game clock, and injuries.
        Also since sports predicting comes with super high variability we anticipate that we will be facing a lot of challenges on accuracy in this exploration. In order to predict a full 62 game March Madness, with a proper accuracy we would need achieve a very high probability prediction. Even with a 90% probability of predicting games accurately, we would still only achieve a .15% probability of predicting all 62 games on the bracket accurately. With a 99% probability

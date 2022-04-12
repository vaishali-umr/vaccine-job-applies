# How Vaccine Requirements Affect Job-Seeker Activity

Given the labor shortage and stiff competition for talent, I was interested in how mention of the COVID vaccine in job postings affects job application activity. I used proprietary data from a database of job ads and candidate click and apply rates, which I had to edit for the purpose of this sample to protect confidential information.

The sample consisted of 380,000 job ads from September and October 2021. The first dataset was the Vaccine Reference Data. There were 2 possible classification tags for the vaccine reference: COVID Vaccine Required and Not Applicable. The latter included posts where the COVID vaccine was not required but encouraged as well as posts where no vaccine was mentioned. The second dataset was the Job Activity Data.

After doing some basic data exploration, I created a new data frame with aggregated job activity so that I could do a high-level comparison. I also made a function to calculate another common metric in the job advertising world – Click to Apply Rate or CTA. This is the conversion rate of clicks to applies and is commonly calculated as the (# paid applies / # paid clicks)*100.

I first created a multiple subplot to show some high-level differences between job-seeker activity for ads where the vaccine was required vs. not. I could see that I had a lot more data for “Not Applicable” job ads than for “COVID Vaccine Required.” Keeping that in mind, listing the vaccine as required led to lower Click to Apply Rates (by 0.31%) than not.

Then I used the industry-level data to show the difference in CTA’s across various industries. The difference in CTA’s was most pronounced in the Customer Service, Farming, Food Service, Military, and Real Estate industries. Not too surprisingly, the Click to Apply Rate in Healthcare was lower when the vaccine was not required; this makes sense since cautious, front-line healthcare workers probably want to work in settings where everyone is vaccinated.

I also created an interactive plot to show click, apply, and CTA data by vaccine requirement. I liked the idea of being able to select the specific industry and metric you want to zone into. Because I generalized my plotting functions, I was able to make similar visualizations for job function data. However, I limited this to my job categories of interest. It was interesting to note that Click to Apply Rates for Waiter and Waitress jobs were over 10% higher when the vaccine was not required – this is what I was expecting. On the other hand, rates were over 10% lower for Customer Service Representative jobs when the vaccine was not required. Hospitality jobs are more often hourly, blue-collar work where we see lower vaccination rates while Customer Service jobs tend to be more corporate work with higher vaccination and education rates. This may start to help explain some of the differences, although more research must be done.

The last step was the statistical analysis. I wanted to try an OLS regression so that I could use the model to predict Click to Apply Rates by vaccine requirement. I turned the vaccine tags into a dummy variable where 0 = COVID Vaccine Required, and 1 = Not Applicable. The low R2 value of 0.023 indicates this is a weak model fit. The p-value of 0.248 says there is a 24.8% chance that the COVID vaccine requirement has no effect on the Click to Apply Rate and that our results are produced by pure chance. The p-value is not smaller than my set alpha of 0.05, so the difference in Click to Apply Rate observed here is NOT statistically significant.

I then decided to add “Count of Jobs” as an additional explanatory variable to see if that improves the model fit.Although the R2 did improve to 0.040, this is still not a strong fit. Very little of the variance in Click to Apply (%) can be explained by vaccine requirement and job count. The new p-values (0.159 and 0.331) are still not smaller than the alpha of 0.05. These observations are still not statistically significant even though the model fit slightly improved with the additional predictor variable.

Prior to this analysis, I thought that listing the COVID vaccine as required would significantly decrease job-seeker activity as anyone might apply to the “Not Applicable” ads but only those who are vaccinated would apply to the “COVID Vaccine Required” postings. While I observed some difference, it was not statistically significant. Some limitations of this analysis are that I used only 2 vaccine tag categories and that there was a big discrepancy in the number of job ads sampled for each vaccine category. For future research, I believe it is worth using the following 3 vaccine categories: Vaccine Required, Vaccine Encouraged (but not required), and Vaccine Not Mentioned. I would also like to sample more job ads in the Vaccine Required category than were used in this analysis.

Due to the poor OLS fit, I think it is also worth exploring other statistical tests like a correlation test or a one-way ANOVA, which can be done if using 3 vaccine categories as mentioned above. A correlation test would not be as descriptive or explanatory as OLS, but it should be a better fit. The ANOVA test could also be interesting because while it will not indicate directionality, it will show any statistical differences between the means of the three vaccine tag groups.

Lastly, I think it would be very interesting to pull data on the geographic locations of each job ad. I am curious to know how geography, along with vaccine mention, affects job-seeker activity. My hunch is that the effect is more pronounced in certain parts of the country. Adding geographic location as an independent variable can be useful for running a two-way ANOVA analysis.

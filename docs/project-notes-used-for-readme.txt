Project: SMS Spam Detection


Dataset:
SMS spam collection


Reason for selection:
- Appropriate size(5572 rows)
- Binary classification problem
- clean structure
- suitable for beginner NLP project.

rows:5572
columns:2

Features:
-message

Target:
-label

Missing Data:
- None

Class Distribution:
- Ham : 4825(86.6%)
- Spam : 747(13.4%)

Duplicate Records:
- 403 duplicate records were detected.

After analysis:
- 309 duplicate ham messages
- 94 duplicate spam messages

Duplicates were removed to reduce bias caused by repeated messages and improve data quality.



Message Length Analysis

A new feature called message_length was created by calculating the number of characters in each SMS message.

Results

- Average spam message length: 138.67 characters
- Average ham message length: 71.48 characters
- Maximum message length: 910 characters
- Minimum message length: 2 characters

Observations

- Spam messages are significantly longer than ham messages on average.
- The average spam message is almost twice as long as a normal message.
- The longest message in the dataset was a ham message with 910 characters, showing that message length alone cannot perfectly distinguish spam from ham messages.
- The shortest messages contained only 2 characters and were all "Ok" messages.
- Message length appears to be a useful feature for spam detection, but it should be combined with other text-based features for better classification performance.
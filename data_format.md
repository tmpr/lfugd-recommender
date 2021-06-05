### Data format clarifications:

    'sampled_1000_items_inter.txt'

#### User-item interaction:


| User Id | Track Id | Number of Interactions | 
| ---    |   ---  |   ---  |
| 0 | 0 | 3  |
| 0 | 6 | 5 |
| 2 | 17 | 8 |
| ... | ... | ... |


    'sampled_1000_items_tracks.txt'
Track-related information (line index, starting from zero, is the **Track ID**):


| Artist | Track Name |
| ---    |   ---  |
| Helstar | Harsh Reality |
| Carpathian Forest | Dypfryst / Dette Er Mitt Helvete |
| Cantique Lépreux | Tourments Des Limbes Glacials |
| ... | ... |


    'sampled_1000_items_demo.txt'
User-related information (line index, starting from zero, is the **User ID**):

| Location | Age | Gender | Reg. Date |
|   ---  |   ---  |   ---  |   ---  |
| RU | 25  | m | 2007-10-12 18:42:00 |
| UK | 27 | m | 2006-11-17 16:51:56 |
| US | 22 | m | 2010-02-02 22:30:15 |
| ... | ... | ... | ... |

All files are in <font color='red'>.tsv (tab '**\t**' separated values)</font> format.
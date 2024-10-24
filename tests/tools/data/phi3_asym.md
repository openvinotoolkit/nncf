| mode                                      | %int4   | %int8   | lora<br>rank   | average<br>relative<br>error   | compression<br>rate   |
|:------------------------------------------|:--------|:--------|:---------------|:-------------------------------|:----------------------|
| fp32                                      | 0%      | 0%      |                | 0.0%                           | 1.0x                  |
| int8                                      | 0%      | 100%    |                | 1.0%                           | 4.0x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 256.0          | 3.9%                           | 6.0x                  |
| int4 + scale estimation                   | 40%     | 60%     |                | 4.1%                           | 4.8x                  |
| int4 + scale estimation                   | 60%     | 40%     |                | 4.3%                           | 5.4x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 128.0          | 4.6%                           | 6.5x                  |
| int4 + scale estimation                   | 80%     | 20%     |                | 5.7%                           | 6.1x                  |
| int4 + scale estimation + lora correction | 100%    | 0%      | 8.0            | 5.8%                           | 7.1x                  |
| int4 + scale estimation + gptq            | 100%    | 0%      |                | 6.1%                           | 7.1x                  |
| int4 + scale estimation                   | 100%    | 0%      |                | 7.5%                           | 7.1x                  |
| int4                                      | 100%    | 0%      |                | 11.9%                          | 7.1x                  |

# game-template

To instantiate a game on replit. (Don't use the inbuilt git integration, it sucks)

Example below is for connect4. Adapt game name

```
git clone https://github.com/Delta-Academy/connect4
mv connect4/* connect4/.* .
rm -r connect4/
```

You should only see main.py and game_mechanics.py. All other files should be hidden by .replit

To push to the repo from replit, don't login globally:

```
git -c "user.name=YOURUSENAME" -c "user.email=YOUREMAIL" commit -m "COMMIT MESSAGE"
```

(add a PAT when prompted for password)

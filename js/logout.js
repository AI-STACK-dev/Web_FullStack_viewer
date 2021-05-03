function init()
{
    logoutWithKakao();
}
function logoutWithKakao(kakaokey)
{
    Kakao.init(kakaokey);
    Kakao.isInitialized();

    if(!Kakao.Auth.getAccessToken()){
        console.log('Not logged in');
        return;
    }
    Kakao.Auth.logout(function(){
       console.log('Not logged in.');
    });

}

init();
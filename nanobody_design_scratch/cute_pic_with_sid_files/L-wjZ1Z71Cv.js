;/*FB_PKG_DELIM*/

__d("PolarisProfileSuggestedUsers_response.graphql",[],(function(a,b,c,d,e,f){"use strict";a={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"PolarisProfileSuggestedUsers_response",selections:[{alias:null,args:null,concreteType:"XDTUserDict",kind:"LinkedField",name:"users",plural:!0,selections:[{alias:null,args:null,concreteType:"XDTRelationshipInfoDict",kind:"LinkedField",name:"friendship_status",plural:!1,selections:[{alias:null,args:null,kind:"ScalarField",name:"following",storageKey:null}],storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"full_name",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"is_verified",storageKey:null},{kind:"RequiredField",field:{alias:null,args:null,kind:"ScalarField",name:"pk",storageKey:null},action:"THROW",path:"users.pk"},{alias:null,args:null,kind:"ScalarField",name:"profile_pic_url",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"username",storageKey:null},{args:null,kind:"FragmentSpread",name:"PolarisFollowChainingList_suggested_users"},{args:null,kind:"FragmentSpread",name:"PolarisProfileUserListDialog_users"}],storageKey:null},{args:null,kind:"FragmentSpread",name:"usePolarisDismissChainingUser_response"}],type:"XDTChainingResponse",abstractKey:null};e.exports=a}),null);
__d("usePolarisDismissChainingUserMutation_instagramRelayOperation",[],(function(a,b,c,d,e,f){e.exports="7705127119553423"}),null);
__d("usePolarisDismissChainingUserMutation.graphql",["usePolarisDismissChainingUserMutation_instagramRelayOperation"],(function(a,b,c,d,e,f){"use strict";a=function(){var a={defaultValue:null,kind:"LocalArgument",name:"chaining_user_id"},c={defaultValue:null,kind:"LocalArgument",name:"target_id"},d=[{alias:null,args:[{kind:"Variable",name:"chaining_user_id",variableName:"chaining_user_id"},{kind:"Variable",name:"target_id",variableName:"target_id"}],concreteType:"XDTChainingDismissResponse",kind:"LinkedField",name:"xdt_api__v1__discover__chaining_dismiss",plural:!1,selections:[{alias:null,args:null,kind:"ScalarField",name:"chaining_user_id",storageKey:null}],storageKey:null}];return{fragment:{argumentDefinitions:[a,c],kind:"Fragment",metadata:null,name:"usePolarisDismissChainingUserMutation",selections:d,type:"Mutation",abstractKey:null},kind:"Request",operation:{argumentDefinitions:[c,a],kind:"Operation",name:"usePolarisDismissChainingUserMutation",selections:d},params:{id:b("usePolarisDismissChainingUserMutation_instagramRelayOperation"),metadata:{is_distillery:!0},name:"usePolarisDismissChainingUserMutation",operationKind:"mutation",text:null}}}();e.exports=a}),null);
__d("usePolarisDismissChainingUser_response.graphql",[],(function(a,b,c,d,e,f){"use strict";a={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"usePolarisDismissChainingUser_response",selections:[{kind:"ClientExtension",selections:[{alias:null,args:null,kind:"ScalarField",name:"__id",storageKey:null}]}],type:"XDTChainingResponse",abstractKey:null};e.exports=a}),null);
__d("usePolarisDismissChainingUser",["CometRelay","FBLogger","Promise","polarisGetXDTUserDict","react","usePolarisDismissChainingUserMutation.graphql","usePolarisDismissChainingUser_response.graphql"],(function(a,b,c,d,e,f,g){"use strict";var h,i,j,k;e=k||d("react");e.useCallback;var l=e.unstable_useMemoCache;function a(a){var e=l(3),f=d("CometRelay").useMutation(h!==void 0?h:h=b("usePolarisDismissChainingUserMutation.graphql")),g=f[0],k=d("CometRelay").useFragment(i!==void 0?i:i=b("usePolarisDismissChainingUser_response.graphql"),a);e[0]!==k.__id||e[1]!==g?(f=function(a,d){var e=function(a){var b,e=c("polarisGetXDTUserDict")(a,d);if(e==null){c("FBLogger")("ig_web").warn("Cannot find user to chaining list");return}b=(b=(b=a.get(k.__id))==null?void 0:(b=b.getLinkedRecords("users"))==null?void 0:b.filter(function(a){return a!==e}))!=null?b:[];(a=a.get(k.__id))==null?void 0:a.setLinkedRecords(b,"users")};e={optimisticUpdater:e,updater:e,variables:{chaining_user_id:d,target_id:a}};g(e);return(j||(j=b("Promise"))).resolve()},e[0]=k.__id,e[1]=g,e[2]=f):f=e[2];return f}g["default"]=a}),98);
__d("PolarisProfileSuggestedUsers.next.react",["fbt","CometRelay","JSResourceForInteraction","PolarisConnectionsLogger","PolarisFollowChainingList.next.react","PolarisProfileSuggestedUsers_response.graphql","XPolarisProfileControllerRouteBuilder","promiseDone","react","useIGDSLazyDialog","usePolarisDismissChainingUser","usePolarisFollowUser","usePolarisIsSmallScreen"],(function(a,b,c,d,e,f,g,h){"use strict";var i,j,k=(j||(j=d("react"))).unstable_useMemoCache,l=j,m=c("JSResourceForInteraction")("PolarisProfileUserListDialog.react").__setRef("PolarisProfileSuggestedUsers.next.react"),n=h._("__JHASH__Ze8enBotJFp__JHASH__");function a(a){var e=k(24),f=a.chainingResponse,g=a.clickPoint,h=a.userID;a=a.username;var j=c("usePolarisIsSmallScreen")();f=d("CometRelay").useFragment(i!==void 0?i:i=b("PolarisProfileSuggestedUsers_response.graphql"),f);var o=c("usePolarisFollowUser")(),p=f.users,q;e[0]!==p?(q=p.map(function(a){var b;return{fullName:a.full_name,id:a.pk,isFollowedByViewer:((b=a.friendship_status)==null?void 0:b.following)===!0,isVerified:a.is_verified,profilePictureUrl:a.profile_pic_url,username:(b=a.username)!=null?b:""}}),e[0]=p,e[1]=q):q=e[1];q=q;var r=c("usePolarisDismissChainingUser")(f);e[2]!==r||e[3]!==h?(f=function(a){c("promiseDone")(r(h,a))},e[2]=r,e[3]=h,e[4]=f):f=e[4];f=f;var s;e[5]!==o?(s=function(a){c("promiseDone")(o(a,!0))},e[5]=o,e[6]=s):s=e[6];s=s;var t;e[7]!==o?(t=function(a){c("promiseDone")(o(a,!1))},e[7]=o,e[8]=t):t=e[8];t=t;var u;e[9]!==a?(u=c("XPolarisProfileControllerRouteBuilder").buildURL({username:a}),e[9]=a,e[10]=u):u=e[10];a=u;u=c("useIGDSLazyDialog")(m);var v=u[0];e[11]!==v||e[12]!==p?(u=function(){v({hideFollowButton:!1,hideName:!1,hideSocialContext:!0,title:n,users:p})},e[11]=v,e[12]=p,e[13]=u):u=e[13];u=u;var w;e[14]!==q||e[15]!==g||e[16]!==j||e[17]!==s||e[18]!==u||e[19]!==f||e[20]!==t||e[21]!==a||e[22]!==p?(w=l.jsx(c("PolarisFollowChainingList.next.react"),{analyticsContext:d("PolarisConnectionsLogger").CONNECTIONS_CONTAINER_MODULES.profile,chainingSuggestions:q,clickPoint:g,impressionModule:d("PolarisConnectionsLogger").VIEW_MODULES.web_profile_chaining,isSmallScreen:j,onFollowUser:s,onSeeAllClick:u,onSuggestionDismissed:f,onUnfollowUser:t,seeAllHref:a,showDescription:!0,title:n,users:p}),e[14]=q,e[15]=g,e[16]=j,e[17]=s,e[18]=u,e[19]=f,e[20]=t,e[21]=a,e[22]=p,e[23]=w):w=e[23];return w}g["default"]=a}),226);
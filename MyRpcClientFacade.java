package creditscoring;
import org.apache.flume.Event;
import org.apache.flume.EventDeliveryException;
import org.apache.flume.api.RpcClient;
import org.apache.flume.api.RpcClientFactory;
import org.apache.flume.event.EventBuilder;
import java.nio.charset.Charset;
/**
 * Created by thinkpad on 2017/12/5.
 */
public class MyRpcClientFacade {
    private RpcClient client;
    private String hostname;
    private int port;
    public void init(String hostname, int port) {
        // 初始化RPC客户端
        this.hostname = hostname;
        this.port = port;
        this.client = RpcClientFactory.getDefaultInstance(hostname, port);
        // 使用下面的方法创建thrift的客户端
        // this.client = RpcClientFactory.getThriftInstance(hostname, port);
    }
    public void sendDataToFlume(String data) {
        // 创建事件对象
        Event event = EventBuilder.withBody(data, Charset.forName("UTF-8"));
        try {// 发送事件
            client.append(event);
        } catch (EventDeliveryException e) {
            // 清除信息，重建Client
            client.close();
            client = null;
            client = RpcClientFactory.getDefaultInstance(hostname, port);
        }
    }
    public void cleanUp() {
        // 关闭RPC连接
        client.close();
    }
}